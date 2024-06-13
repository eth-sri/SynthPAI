import json
import os
import hashlib

def comment_to_hex(comment, size=50):
    # Compute hash of comment text
    text = comment["text"].strip()
    length = min(size, len(text))
    prefix = text[:length] + comment["username"]

    hash_object = hashlib.sha256(prefix.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig

def generate_thread_comments(comments):

    comments_hex_to_id = {}
    comments_hex_to_id_short = {}
    comments_id_to_hex = {}
    for comment in comments.values():
        # Compute hash of comment text
        comments_hex_to_id[comment_to_hex(comment)] = comment["id"]
        comments_hex_to_id_short[comment_to_hex(comment, size=20)] = comment["id"]
        comments_id_to_hex[comment["id"]] = comment_to_hex(comment)

    def walk_thread(comment, thread_id, parent_id=None):

        if parent_id is not None:
            hex = comment_to_hex(comment)

            if hex not in comments_hex_to_id:
                print(
                    f"Comment not found: {comment['text'][:100]}"
                )  # Intermediary comment not found

                hex_short = comment_to_hex(comment, size=20)
                if hex_short not in comments_hex_to_id_short:
                    # We skip this comment
                    id = parent_id
                else:
                    # Fallback for short modified comments
                    id = comments_hex_to_id_short[hex_short]
                    comment_obj = comments[id]

                    comment_obj["parent_id"] = parent_id
                    comment_obj["thread_id"] = thread_id

                    if "children" not in comment_obj:
                        comment_obj["children"] = []

            else:

                id = comments_hex_to_id[hex]
                comment_obj = comments[id]

                comment_obj["parent_id"] = parent_id
                comment_obj["thread_id"] = thread_id

                if "children" not in comment_obj:
                    comment_obj["children"] = []

            if parent_id != "root":
                parent_obj = comments[parent_id]
                parent_obj["children"].append(id)

        else:
            id = "root"
            try:
                comment_obj = comments[comments_hex_to_id[comment_to_hex(comment)]]
                comment_obj["thread_id"] = thread_id
                comment_obj["parent_id"] = None
                id = comments_hex_to_id[comment_to_hex(comment)]
                comment_obj["children"] = []
            except KeyError:
                print(f"Root Comment not found: {comment['text'][:100]}")

        # Set children
        if "children" in comment:
            for child in comment["children"]:
                walk_thread(child, thread_id, id)

    thread_folder = "data/thread/generated_threads/json_threads"
    # Iterate through all files subfolders
    for root, dirs, files in os.walk(thread_folder):
        print(f"Processing file: {files}")
        for file in files:
            if file.endswith(".json"):
                thread_path = os.path.join(root, file)
                with open(thread_path, "r") as file:
                    data = json.loads(file.readlines()[0])
                    thread_id = thread_path.split("/")[-1].split(".")[0]

                    walk_thread(data, thread_id)

if __name__ == "__main__":

    comments = {}
    authors = {}

    with open("data/thread/synth_clean.jsonl", "r") as file:
        for line in file:
            comment = json.loads(line)

            id = comment["id"]
            author = comment["author"]

            comments[id] = comment
            if author not in authors:
                authors[author] = {
                    "username": comment["username"],
                    "profile": comment["profile"],
                    "comment_ids": [id],
                }
            else:
                authors[author]["comment_ids"].append(id)

    generate_thread_comments(comments)

    ctr = 0
    for comment in comments.values():
        if "thread_id" not in comment:
            ctr += 1
            print(
                f"Thread not found for comment: {comment['text'][:100].strip()} - {comment['id']}"
            )

    print(f"Number of comments without thread: {ctr}")

    # Store updated comments
    with open("data/synthpai.jsonl", "w") as file:
        for comment in comments.values():
            file.write(json.dumps(comment) + "\n")
            
    # convert age guesses to str format
    with open('data/synthpai.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]

    for item in data:
        if item.get('guesses') is not None:
            for pred in item['guesses']:
                if pred.get('feature') == 'age':
                    pred['guesses'] = list(map(str, pred['guesses']))

    with open('data/synthpai.jsonl', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')