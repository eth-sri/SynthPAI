python src/utils/merge.py --files data/thread/eval/gpt-4/gpt4_evaluated.jsonl \
                                    data/thread/eval/gpt-3.5/gpt3.5_evaluated.jsonl \
                                    data/thread/eval/Claude-3-Opus/claude-3-opus_evaluated.jsonl \
                                    data/thread/eval/Claude-3-Haiku/claude-3-haiku_evaluated.jsonl \
                                    data/thread/eval/Claude-3-Sonnet/claude-3-sonnet_evaluated.jsonl \
                                    data/thread/eval/Gemini-Pro/gemini_pro_evaluated.jsonl \
                                    data/thread/eval/Gemini_Pro_1.5/gemini_pro_1.5_evaluated.jsonl \
                                    data/thread/eval/Gemma-7B-Instruct/gemma_7b_evaluated.jsonl \
                                    data/thread/eval/Llama-2-7b/llama2-7b-chathf_evaluated.jsonl \
                                    data/thread/eval/Llama-2-13b/llama2-13b-chathf_evaluated.jsonl \
                                    data/thread/eval/Llama-2-70b/llama2-70b-chathf_evaluated.jsonl \
                                    data/thread/eval/Llama-3-8b/llama3-8b-chathf_evaluated.jsonl \
                                    data/thread/eval/Llama-3-70b/llama3-70b-chathf_evaluated.jsonl \
                                    data/thread/eval/Mistral-7B-Instruct-v0.1/mistral_7b_evaluated.jsonl \
                                    data/thread/eval/Mixtral-8x7B-Instruct-v0.1/mixtral_8x7b_evaluated.jsonl \
                                    data/thread/eval/Mixtral-8x22B-Instruct-v0.1/mixtral_8x22b_evaluated.jsonl \
                                    data/thread/eval/Qwen1.5-110B-Chat/qwen1.5_110b_evaluated.jsonl \
                                    data/thread/eval/Yi-34B-Chat/yi_34b_evaluated.jsonl \
                                    --outpath data/synthpai_merged_evals.jsonl --merge_key username