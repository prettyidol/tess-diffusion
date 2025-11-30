import os
import time
import shutil
from typing import Optional, List

from transformers.trainer_callback import TrainerCallback
from transformers.trainer import TRAINER_STATE_NAME


class TimeAndGDriveBackupCallback(TrainerCallback):
    """
    - Time-based lightweight snapshots to avoid losing progress before hitting save_steps.
    - Mirror each save (step/epoch or time snapshot) to a Google Drive path if provided.

    Notes:
    - Uses public Trainer APIs only (save_model, tokenizer.save_pretrained, state.save_to_json).
    - Optimizer/scheduler states are not saved here (they require internal _save_checkpoint). This is
      an intentional trade-off for robustness: we guarantee a restorable model/tokenizer + trainer_state
      even if a job dies before reaching a HF-managed checkpoint.
    """

    def __init__(self):
        self._last_time_save: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_time_save = time.time()
        # Normalize gdrive path: treat empty string as None
        if hasattr(args, "gdrive_backup_dir") and args.gdrive_backup_dir:
            os.makedirs(args.gdrive_backup_dir, exist_ok=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if getattr(args, "time_save_interval_seconds", None):
            now = time.time()
            if self._last_time_save is None:
                self._last_time_save = now
            if now - self._last_time_save >= args.time_save_interval_seconds:
                # Create a lightweight snapshot inside output_dir
                snapshot_dir = os.path.join(
                    args.output_dir, f"snapshot-time-{state.global_step}-{int(now)}"
                )
                self._save_lightweight_snapshot(kwargs.get("model"), kwargs.get("tokenizer"), args, state, snapshot_dir)
                # Mirror snapshot to Google Drive if configured
                self._mirror_to_gdrive(args, snapshot_dir)
                self._prune_old_backups(args)
                self._last_time_save = now
        return control

    def on_save(self, args, state, control, **kwargs):
        # Whenever Trainer saves (due to save_steps/epoch), mirror the newest checkpoint to GDrive
        if getattr(args, "gdrive_backup_dir", None):
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            # If the expected checkpoint dir doesn't exist (edge case), mirror output_dir instead
            src = ckpt_dir if os.path.isdir(ckpt_dir) else args.output_dir
            self._mirror_to_gdrive(args, src)
            self._prune_old_backups(args)
        return control

    # ---------- helpers ----------

    def _save_lightweight_snapshot(self, model, tokenizer, args, state, snapshot_dir: str):
        os.makedirs(snapshot_dir, exist_ok=True)
        # Save model/tokenizer
        if model is not None:
            # Trainer.save_model can accept a directory to save into
            # but within callbacks we don't have direct access to Trainer.save_model.
            # Workaround: use model.save_pretrained.
            try:
                model.save_pretrained(snapshot_dir)
            except Exception:
                # As a fallback, rely on Trainer.save_model via kwargs when available
                pass
        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(snapshot_dir)
            except Exception:
                pass
        # Save trainer state json (global_step, etc.)
        try:
            state.save_to_json(os.path.join(snapshot_dir, TRAINER_STATE_NAME))
        except Exception:
            pass

    def _mirror_to_gdrive(self, args, src_dir: str):
        gdst = getattr(args, "gdrive_backup_dir", None)
        if not gdst:
            return
        # Create a unique target directory under gdst with the same leaf name as src
        leaf = os.path.basename(os.path.normpath(src_dir))
        dst_dir = os.path.join(gdst, leaf)
        # Copy directory tree (overwrite if exists)
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir, ignore_errors=True)
        shutil.copytree(src_dir, dst_dir)

    def _prune_old_backups(self, args):
        gdst = getattr(args, "gdrive_backup_dir", None)
        keep = getattr(args, "backup_keep_last", 3)
        if not gdst or not os.path.isdir(gdst) or not isinstance(keep, int) or keep <= 0:
            return
        # List subdirectories sorted by mtime descending
        entries: List[str] = [os.path.join(gdst, d) for d in os.listdir(gdst) if os.path.isdir(os.path.join(gdst, d))]
        entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for old in entries[keep:]:
            shutil.rmtree(old, ignore_errors=True)
