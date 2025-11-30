import os
import psutil
import time
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import spacy
import re

# ================================
# 1️⃣ 基本路径配置
# ================================
DATA_DIR = "/mnt/d/idol01/homework/paper_code/tess-diffusion/processed_data/openwebtext_50/train"
OUTPUT_DIR = "/mnt/d/idol01/homework/paper_code/tess-diffusion/four"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# 2️⃣ 模型占位符 (将在每个工作进程中初始化)
# ================================
rebel = None
rebel_model = None
rebel_tok = None
tokenizer = None
nlp = None

def init_worker():
    """在每个工作进程中独立初始化模型，避免多进程冲突。"""
    global rebel, tokenizer, nlp, rebel_model, rebel_tok  # 提前声明所有将被赋值的全局变量

    # 检查是否已初始化，防止重复加载
    if rebel_model is None or rebel_tok is None or tokenizer is None or nlp is None:
        print(f"🚀 Initializing models in worker process: {os.getpid()}")
        # RoBERTa 解码器（用于把 input_ids -> text）
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        # REBEL 原生模型与分词器（保留特殊 token）
        model_name = "Babelscape/rebel-large"
        rebel_tok = AutoTokenizer.from_pretrained(model_name)
        rebel_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        rebel_model.eval()

        # spaCy
        nlp = spacy.load("en_core_web_sm")

        # 选择设备
        device = "cpu"
        rebel_model.to(device)

# ================================
# 3️⃣ 时间提取辅助函数
# ================================
def extract_time_expressions(text):
    doc = nlp(text)
    times = [ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]
    regex_times = re.findall(r"\b(?:\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{0,2},?\s*\d{0,4})\b", text)
    times.extend(regex_times)
    return list(set(times)) if times else ["N/A"]

# ================================
# 3.1️⃣ 解析 REBEL 生成结果的工具函数（适配多种格式）
# ================================
def parse_rebel_output(generated_text: str):
    """将 REBEL 的生成文本解析为 [(head, relation, tail), ...]。
    兼容以下几种常见格式：
    - 带特殊标记：<triplet> subject <subj> relation <rel> object <obj>
    - 带分隔符：<triplet> subject <sep> relation <sep> object </triplet>
    - 旧版括号格式：(head, relation, tail)
    - 兜底：根据双空格启发式分块，按 (ent, rel, ent) 三元组滑窗提取（弱保证）
    """
    triples = []
    s = generated_text or ""
    s = s.strip()
    if not s:
        return triples

    # 优先：<triplet> ... 解析
    if "<triplet>" in s:
        parts = s.split("<triplet>")
        for p in parts[1:]:
            # 截断至 </triplet>（如果存在）
            if "</triplet>" in p:
                p = p.split("</triplet>")[0]
            # 去掉句界标记
            if "</s>" in p:
                p = p.split("</s>")[0]
            p = p.replace("<s>", " ").strip()
            # 三种标记尝试
            if all(tag in p for tag in ["<subj>", "<rel>", "<obj>"]):
                try:
                    i_sub = p.index("<subj>")
                    i_rel = p.index("<rel>")
                    i_obj = p.index("<obj>")
                    head = p[:i_sub].strip()
                    rel = p[i_sub + len("<subj>"): i_rel].strip()
                    tail = p[i_rel + len("<rel>"): i_obj].strip()
                    if head and rel and tail:
                        triples.append((head, rel, tail))
                    continue
                except Exception:
                    pass
            # 常见格式：只出现 <subj> 与 <obj>，关系出现在 <obj> 之后
            if ("<subj>" in p) and ("<obj>" in p):
                try:
                    i_sub = p.index("<subj>")
                    i_obj = p.index("<obj>")
                    head = p[:i_sub].strip()
                    tail = p[i_sub + len("<subj>"): i_obj].strip()
                    rel = p[i_obj + len("<obj>"):].strip()
                    # 清理多余空白
                    head = head.strip(" -:;,.\t\n")
                    tail = tail.strip(" -:;,.\t\n")
                    rel = rel.strip(" -:;,.\t\n")
                    # 排除明显无效的关系（空、仅数字等）
                    if head and tail and rel and not re.fullmatch(r"\d{1,4}", rel):
                        triples.append((head, rel, tail))
                        continue
                except Exception:
                    pass
            # <sep> 分隔
            if "<sep>" in p:
                fields = [x.strip() for x in p.split("<sep>") if x.strip()]
                if len(fields) >= 3:
                    head, rel, tail = fields[:3]
                    triples.append((head, rel, tail))
                    continue
            # 兜底：尝试用括号格式或启发式
            # 后续会统一处理

    # 括号 (h, r, t)
    if "(" in s and ")" in s:
        for chunk in s.split("("):
            if ")" in chunk:
                body = chunk.split(")")[0]
                parts = [x.strip() for x in body.split(",")]
                if len(parts) == 3 and all(parts):
                    triples.append(tuple(parts))

    # 如果仍为空，启发式：按双空格切块，滑窗取 (ent, rel, ent)
    if not triples:
        tokens = [t.strip() for t in s.split("  ") if t.strip()]
        # 采用长度为3的滑动窗口，要求中间块像是关系短语（包含空格或小写词）
        for i in range(0, len(tokens) - 2):
            h, r, t = tokens[i], tokens[i + 1], tokens[i + 2]
            # 简单约束：关系通常非全大写，且包含空格/为多词
            if r and (" " in r or r.islower()) and h and t:
                # 排除看起来像年份的中间块
                if not re.fullmatch(r"\d{1,4}", r):
                    triples.append((h, r, t))
    return triples

# ================================
# 3.2️⃣ 使用 REBEL 原生模型生成（保留特殊token）
# ================================
@torch.no_grad()
def rebel_generate(text: str, max_length: int = 384, device: str = "cpu") -> str:
    """用 REBEL 原生模型生成，禁用 skip_special_tokens 以保留标记，便于解析。"""
    if rebel_model is None or rebel_tok is None:
        init_worker()
    inputs = rebel_tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    if device == "cuda" and torch.cuda.is_available():
        model_device = torch.device("cuda")
    else:
        model_device = torch.device("cpu")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    rebel_model.to(model_device)
    outputs = rebel_model.generate(**inputs, max_length=max_length)
    decoded = rebel_tok.batch_decode(outputs, skip_special_tokens=False)[0]
    return decoded
# ================================
# 4️⃣ REBEL + 时间抽取函数 (包含解码逻辑)
# ================================
def extract_relations_with_time(example):
    # 确保模型已在当前进程中初始化
    init_worker()

    text = ""
    # 检查并解码 input_ids
    if "input_ids" in example:
        text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
    elif "text" in example: # 保留对纯文本格式的兼容
        text = example["text"]
    
    if not text.strip():
        return {"quadruples": []}

    try:
        # 截取部分文本以提高效率
        text_snippet = text[:512]
        # 使用原生模型生成，保留特殊token
        generated_text = rebel_generate(text_snippet, max_length=384)

        # 解析生成结果为 (h, r, t)
        triples = parse_rebel_output(generated_text)

        # 提取时间并组装四元组
        times = extract_time_expressions(text_snippet)
        quadruples = [(h, r, t, times[0]) for (h, r, t) in triples]

        # 可选：调试前几条
        if not hasattr(extract_relations_with_time, "_dbg"): 
            extract_relations_with_time._dbg = 0
        extract_relations_with_time._dbg += 1
        if extract_relations_with_time._dbg <= 3:
            print("\n🔎 DEBUG sample:")
            print("Text:", text_snippet[:120].replace("\n", " "), "...")
            print("Generated:", generated_text[:200].replace("\n", " "), "...")
            print("Triples:", triples[:3])
            print("Times:", times[:3])

        return {"quadruples": quadruples}
    except Exception as e:
        # 打印错误以便调试
        print(f"Error in process {os.getpid()}: {e}")
        return {"quadruples": [], "error": str(e)}

# ================================
# 5️⃣ 内存检测函数
# ================================
def wait_for_memory(threshold_gb=2):
    while True:
        free_mem = psutil.virtual_memory().available / (1024**3)
        if free_mem < threshold_gb:
            print(f"⚠️ Low memory ({free_mem:.2f} GB free). Waiting 10s...")
            time.sleep(10)
        else:
            break

# ================================
# 6️⃣ 主批处理逻辑
# ================================
if __name__ == "__main__":
    arrow_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".arrow")])
    print(f"🔹 Found {len(arrow_files)} Arrow files.")

    # 本轮目标：对 17..91 号分片，在已有 ~50 样本的基础上，追加样本使每个CSV达到 200 行
    UPDATE_START, UPDATE_END = 17, 91  # 包含端点
    TARGET_SAMPLES = 200
    BASE_OFFSET = 50                  # 期望在已有50基础上追加

    # 配置：处理规模与写盘策略（不压榨性能）
    NUM_PROC = 4                 # 并行进程数（保持不变）
    FLUSH_EVERY_N = 20000        # 累积到N行即追加写盘，避免占内存

    def csv_has_rows(path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            df_head = pd.read_csv(path, nrows=1)
            return len(df_head) > 0
        except Exception:
            return False

    def csv_row_count(path: str) -> int:
        """读取CSV行数（不含表头）。不存在则为0。"""
        if not os.path.exists(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            return 0

    for i, file in enumerate(arrow_files):
        base_name = os.path.basename(file).replace(".arrow", "")
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}_quads.csv")

        # 解析分片编号（例如 data-00020-of-00092 -> 20）
        try:
            shard_idx = int(base_name.split("-")[1])
        except Exception:
            print(f"⚠️  Cannot parse shard index from {base_name}, skipping.")
            continue

        # 仅处理 17..91 范围
        if not (UPDATE_START <= shard_idx <= UPDATE_END):
            continue

        # 计算当前CSV已有行数，决定追加区间
        existing_rows = csv_row_count(output_file)
        if existing_rows >= TARGET_SAMPLES:
            print(f"✅ Skipping {base_name} (already has {existing_rows} rows ≥ {TARGET_SAMPLES}).")
            continue
        # 从50开始追加，如果不足50则从已有行数开始，目标到200
        start_idx = BASE_OFFSET if existing_rows >= BASE_OFFSET else existing_rows
        end_idx = TARGET_SAMPLES
        print(f"\nProcessing shard {shard_idx}: {base_name} -> append [{start_idx}:{end_idx}) (current={existing_rows})")
        wait_for_memory(4) # 建议为多进程留出更多内存

        try:
            # 使用流式处理，不将数据完全加载到内存
            dataset = load_dataset("arrow", data_files=file, cache_dir=None, keep_in_memory=False)["train"]

            total = len(dataset)
            if start_idx >= total:
                print(f"ℹ️  start_idx {start_idx} exceeds total {total}, skip {base_name}")
                continue
            end_eff = min(end_idx, total)
            if start_idx >= end_eff:
                print(f"ℹ️  Nothing to do for {base_name} (start {start_idx} >= end {end_eff}).")
                continue
            dataset = dataset.select(range(start_idx, end_eff))
            print(f"📊 Appending samples [{start_idx}:{end_eff}) out of {total}")

            dataset_with_quads = dataset.map(
                extract_relations_with_time,
                num_proc=NUM_PROC,
                batched=False
            )

            # 增量写盘
            rows = []
            # 如果文件已存在且有任意内容，则不再写表头
            wrote_header = os.path.exists(output_file) and os.path.getsize(output_file) > 0
            for example in tqdm(dataset_with_quads, desc=f"Saving {base_name} (append)"):
                for quad in example["quadruples"]:
                    if len(quad) == 4:
                        rows.append({
                            "head": quad[0],
                            "relation": quad[1],
                            "tail": quad[2],
                            "time": quad[3]
                        })
                if len(rows) >= FLUSH_EVERY_N:
                    df = pd.DataFrame(rows)
                    df.to_csv(output_file, mode="a", index=False, header=not wrote_header)
                    wrote_header = True
                    rows.clear()

            # flush 最后一批
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(output_file, mode="a", index=False, header=not wrote_header)
                rows.clear()

            # 统计行数
            try:
                saved_rows = csv_row_count(output_file)
                print(f"✅ Now {saved_rows} rows in {output_file}")
            except Exception:
                print(f"✅ Saved to {output_file}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
            continue

    print("\n🎉 All Arrow files processed successfully!")