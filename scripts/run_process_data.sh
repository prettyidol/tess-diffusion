# Process (group to fixed length chunks, and tokenize) a dataset.

# HF_HOME="${LOCAL_DIR}/.cache/huggingface/"
HF_HOME="D:/hf_cache" 

# CUDA_VISIBLE_DEVICES="0" 
HF_HOME=$HF_HOME python sdlm/data/process_data.py $1
