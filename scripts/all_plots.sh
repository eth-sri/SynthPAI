# # All baseline plots
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder full --plot_stack --complete
# # HArdness and less precise
# python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder less_precise_plots --plot_hardness --show_less_precise --min_certainty 3

python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder hardness --plot_hardness --model GPT-4 Claude-3-Opus Llama-3-70b  Mixtral-8x22B Qwen1.5-110B

# # Drop plot
python src/visualization/visualize_reddit.py --path data/thread/eval/gpt-4/gpt4_evaluated.jsonl data/thread/eval/gpt-4/anonymized/gpt4_anon_evaluated.jsonl --plot_drop --model GPT-4 --folder gpt4_drop

# Attribute accuracy plots
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model GPT-4
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model GPT-3.5
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Llama-2-7b
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Llama-2-13b
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Llama-2-70b
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Gemma-7B
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Mistral-7B
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Mixtral-8x7B
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Mixtral-8x22B
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Llama-3-8b
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Llama-3-70b
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Yi-34B
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Claude-3-Haiku
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Claude-3-Sonnet
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Claude-3-Opus
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Qwen1.5-110B
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Gemini-Pro
python src/visualization/visualize_reddit.py --path data/synthpai_merged_evals.jsonl --folder attributes --plot_attributes --model Gemini-1.5-Pro
