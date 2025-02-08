# Norwegian Reasoning Processing Pipeline

This repository provides the files for creating a reasoning dataset. The input to the pipeline is here the Wikipedia Paragraph dataset.

1. **Download Cleaned Wikipedia Paragraphs**  
   `wget https://huggingface.co/datasets/pere/wiki_paragraphs_norwegian/raw/main/pretrain.jsonl` downloads 10k Wikipedia paragraphs.

2. **Fetch DeepSeek Reasoning Data**  
   `python fetch_deepseek_reasoning_data.py --input_file pretrain.jsonl` fetched reasoning data from DeepSeek and stores them in `pretrain_procssed.jsonl`. You can edit or change the template file used for this prompt.

3. **Build Prompt**  
   `python build_prompt.py --input_file pretrain_processed.jsonl --output_file pretrain_prompt.jsonl` builds the prompt and adds it to the jsonlines file. You can edit or change the template file used for this prompt.

4. **Create Splits and Upload Dataset**  
   `python create_splits_and_upload_reason_dataset.py --input_file pretrain_prompt.jsonl --repo_id user/repo` creates splits and uploads the file to HuggingFace. You will need to set the sizes for test and validation in the script.

5. **Create MaxText Training Set**
   This is an optional step. 
   - Clone the newly created repo
   - `mkdir shards`
   - `split -n l/16 -d --additional-suffix=.jsonl train.jsonl shards/train_part_`
   - `split -n l/16 -d --additional-suffix=.jsonl validation.jsonl shards/validation_part_`
   - `split -n l/16 -d --additional-suffix=.jsonl test.jsonl shards/test_part_`
   - `gsutil -m cp *.* gs://[mybucket]/wiki_reasoning_english/`
