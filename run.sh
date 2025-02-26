#!/usr/bin/bash

python src/yue/infer_all.py --stage1_use_exl2 --stage2_use_exl2 \
    --stage1_cuda_idx 0 --stage2_cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ./prompt_egs/genre1.txt \
    --lyrics_txt ./prompt_egs/lyrics1.txt \
    --run_n_segments 4 \
    --stage2_batch_size 32 \
    --stage1_cache_size 32768 \
    --stage2_cache_size 32768 \
    --output_dir ./output \
    --max_new_tokens 2048 \
    --repetition_penalty 1.1
