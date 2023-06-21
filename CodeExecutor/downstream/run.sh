source_lang=python
target_lang=python
python run.py \
    --model_name_or_path microsoft/unixcoder-base  \
    --query_data_file ../data/code_to_code_search_test.json \
    --candidate_data_file ../data/code_to_code_search_test.json \
    --trace_file ../saved_models/code_to_code_search/preds.txt \
    --query_lang ${source_lang} \
    --candidate_lang ${target_lang} \
    --code_length 512 \
    --eval_batch_size 256 
