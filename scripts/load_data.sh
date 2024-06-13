# load data from HuggingFace or locally (specify in --file_type parameter: 'HF' for HuggingFace and 'local' for locally)
python data/load_data.py --file_type 'local'

# prepare human labeled data for model inference
python src/thread/prepare_data_for_eval.py