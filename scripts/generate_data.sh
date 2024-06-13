# generate threads
python main.py --config_path configs/thread/thread.yaml

# collect data into one file
python src/thread/collect_data.py

# print generated threads in .txt format
python src/visualization/print_threads.py