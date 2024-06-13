# prepare data for evaluations
python src/thread/label_eval/prepare_data_for_label_eval.py

# run manual checks
python main.py --config_path configs/eval/eval_labels.yaml

# prepare checked dataset for model inference
python src/thread/label_eval/rewrite_human_labels.py