# GPT-4 predict+eval
python main.py --config_path configs/eval/openai/gpt4_predict.yaml
python main.py --config_path configs/eval/openai/gpt4_eval.yaml

# GPT-3.5 predict+eval+prepare output for eval
python main.py --config_path configs/eval/openai/gpt_3.5_predict.yaml
python src/thread/normalize.py --in_paths data/thread/predicted/gpt-3.5/gpt3.5_predicted.jsonl --outpath data/thread/predicted/gpt-3.5/gpt3.5_predicted_fixed.jsonl --fix
python main.py --config_path configs/eval/openai/gpt_3.5_eval.yaml

# Llama-3-70b predict+eval+prepare output for eval
python main.py --config_path configs/eval/meta/llama3_70b_predict.yaml
python src/thread/normalize.py --in_paths data/thread/predicted/Llama-3-70b/llama3-70b-chathf_predicted.jsonl --outpath data/thread/predicted/Llama-3-70b/llama3-70b-chathf_predicted_fixed.jsonl --fix
python main.py --config_path configs/eval/meta/llama3_70b_eval.yaml

# Llama-3-8b predict+eval+prepare output for eval
python main.py --config_path configs/eval/meta/llama3_8b_predict.yaml
python src/thread/normalize.py --in_paths data/thread/predicted/Llama-3-8b/llama3-8b-chathf_predicted.jsonl --outpath data/thread/predicted/Llama-3-8b/llama3-8b-chathf_predicted_fixed.jsonl --fix
python main.py --config_path configs/eval/meta/llama3_8b_eval.yaml

# Llama-2-70b predict+eval+prepare output for eval
python main.py --config_path configs/eval/meta/llama2_70b_predict.yaml
python src/thread/normalize.py --in_paths data/thread/predicted/Llama-2-70b/llama2-70b-chathf_predicted.jsonl --outpath data/thread/predicted/Llama-2-70b/llama2-70b-chathf_predicted_fixed.jsonl --fix
python main.py --config_path configs/eval/meta/llama2_70b_eval.yaml

# Claude-3-Opus predict+eval+prepare output for eval
python main.py --config_path configs/eval/anthropic/claude3_opus_predict.yaml
python main.py --config_path configs/eval/anthropic/claude3_opus_eval.yaml

# Gemini-Pro-1.5 predict+eval+prepare output for eval
python main.py --config_path configs/eval/google/gemini_pro_15_predict.yaml
python src/thread/normalize.py --in_paths data/thread/predicted/Gemini_Pro_1.5/gemini_pro_1.5_predicted.jsonl --outpath data/thread/predicted/Gemini_Pro_1.5/gemini_pro_1.5_predicted_fixed.jsonl --fix
python main.py --config_path configs/eval/google/gemini_pro_15_eval.yaml

# Mixtral-8x22b predict+eval+prepare output for eval
python main.py --config_path configs/eval/mistralai/mixtral_8x22b_instruct_predict.yaml
python src/thread/normalize.py --in_paths data/thread/predicted/Mixtral-8x22B-Instruct-v0.1/mixtral_8x22b_predicted.jsonl --outpath data/thread/predicted/Mixtral-8x22B-Instruct-v0.1/mixtral_8x22b_predicted_fixed.jsonl --fix
python main.py --config_path configs/eval/mistralai/mixtral_8x22b_instruct_eval.yaml

# Qwen-1.5-110b predict+eval+prepare output for eval
python main.py --config_path configs/eval/together/qwen1.5_110b_predict.yaml
python main.py --config_path configs/eval/together/qwen1.5_110b_eval.yaml