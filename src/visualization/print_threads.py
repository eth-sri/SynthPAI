import json
import os

def print_single_thread(node, f, all_data, visited, depth=0):
    
    indent = "    " * depth
    comment = node['text']
    author = node['username']
    print('\n', file=f)
    print(f"{indent} {author}: {comment}", file=f)

    if visited is None:
        visited = []
    visited.append(node['id'])
    
    for child in node['children']:
        for cm in all_data:
            if cm['id'] == child and cm["id"] not in visited:
                print_single_thread(cm, f, all_data, visited, depth + 1)

def print_threads():
    
    features = ["city_country", "age", "income_level", "income", "education", "occupation", "sex", "relationship_status", "birth_city_country"]
    for topic in features:
        for i in range(50):
            filename = "data/synthpai.jsonl"
            if os.path.isfile(filename):
                with open(filename, 'r', encoding="utf-8", errors='ignore') as json_file:
                    thread_data = [json.loads(line) for line in json_file]
                    thread_name = f"thread_{topic}_{i}"
                    for comment in thread_data:
                        if comment["parent_id"] is None:
                            if comment["thread_id"] == thread_name:
                                with open(f"data/thread/generated_threads/txt_threads/{topic}/thread_{topic}_{i}.txt", "w") as f:
                                    print(f'##### Thread topic "{topic}" no.{i} #####', file=f)
                                    print_single_thread(comment, f, thread_data, None)

if __name__ == "__main__":
    print_threads()