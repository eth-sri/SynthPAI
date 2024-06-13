import gradio as gr
import json
import argparse
import os
import time
import random
import tqdm
import json 
import duckdb
import os

def load_data(path, args):
    schema_order = ["author", "num_comments", "comments", "subreddits", "timestamps"]

    extension = path.split(".")[-1]
    data = []
    if extension == "json":
        with open(path) as path:
            data = [json.load(path)]
    elif extension == "jsonl":
        with open(path, "r") as json_file:
            json_list = json_file.readlines()

        for json_str in json_list:
            data.append(json.loads(json_str))
    elif extension == "db" or extension == "duckdb":
        con = duckdb.connect(path)
        con.execute("SET threads TO 1")
        con.execute("SET enable_progress_bar=true")
        data = con.execute(
            f"SELECT * FROM {args.table} USING SAMPLE reservoir(50000 ROWS) REPEATABLE ({args.seed})"
        ).fetchall()

        new_data = []
        for row in data:
            new_data.append({k: v for k, v in zip(schema_order, row)})
        data = new_data

    return data



def write_data(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


data = []
seed = 0
user = ""
timestamp = 0
data_idx = 0
outpath = ""
pbar = None
attributes = [
    "city_country",
    "sex",
    "age",
    "occupation",
    "birth_city_country",
    "relationship_status",
    "income_level",
    "education",
]


def get_new_datapoint_idx():
    global data
    global data_idx
    global pbar

    while data_idx < len(data) - 1:
        new_point = data[data_idx]
        is_valid = True

        ret_data_idx = data_idx
        # Increment the data index
        data_idx += 1
        pbar.update(1)

        if is_valid:
            # Save the current state
            with open(".synth_temp", "w") as f:
                f.write(str(seed) + "\n")
                f.write(str(ret_data_idx) + "\n")

            return ret_data_idx

    print("No more data points")
    exit(1)


def save_and_load_new(full_state_box, *args):
    global out_path
    global data
    global user
    global timestamp
    global data_idx

    # Write out current data point
    if len(full_state_box) > 0:
        assert len(args) == 3 * len(attributes)

        curr_data_point = json.loads(full_state_box)

        has_info = False

        curr_reviews = {}
        if "reviews" in curr_data_point:
            curr_reviews = curr_data_point["reviews"]

        if user in curr_reviews:
            print("WARNING - OVERWRITING PREVIOUS REVIEW")
        curr_reviews[user] = {}

        for i, attr in enumerate(attributes):
            estimated_val = args[i]
            hardness_val = args[i + len(attributes)]
            certainty_val = args[i + 2 * len(attributes)]

            if hardness_val == -1:
                estimated_val == ""
            if estimated_val is None:
                estimated_val = ""

            curr_reviews[user][attr] = {
                "estimate": estimated_val.strip().lower(),
                "hardness": hardness_val,
                "certainty": certainty_val,
            }

        curr_reviews[user]["timestamp"] = time.time()
        curr_reviews[user]["time"] = time.time() - timestamp
        curr_data_point["reviews"] = curr_reviews

        write_data(out_path, curr_data_point)

    # Load new data point
    new_idx = get_new_datapoint_idx()
    new_point = data[new_idx]

    formatted_comments = new_point["text"]

    full_json = json.dumps(new_point)

    # Set the timestamp to meaure how long grading takes
    timestamp = time.time()

    # Set the relevant pre-filled values

    pre_filled_values = [""] * len(attributes)
    hardness_values = [0, 0, 0, 0, 0, 0, 0, 0]
    certainty_values = [0, 0, 0, 0, 0, 0, 0, 0]
    for i, attr in enumerate(attributes):
        if new_point["guesses"] is None:
            continue
        if type(new_point["guesses"]) == str:
            new_point["guesses"] = eval(new_point["guesses"])
        for guess in new_point["guesses"]:
            if guess["feature"] == attr:
                pre_filled_values[i] = str(guess["guesses"][0])
                certainty_values[i] = int(guess["certainty"])

    return (
        formatted_comments,
        full_json,
        *pre_filled_values,
        *hardness_values,
        *certainty_values,
    )


def main(data, args):
    with gr.Blocks() as demo:
        with gr.Row(equal_height=True):
            name = gr.Textbox(label="Posts", value="")

        inputs = []
        dfs_boxes = []
        hardness_sliders = []
        certainty_sliders = []

        with gr.Row(equal_height=True):
            for i, attr in enumerate(attributes):
                with gr.Column():
                    if attr == "city_country":
                        input = gr.Textbox(
                            label="Location (City / State / Country)",
                            value="",
                            info="Give the closest city / state / country",
                        )
                    elif attr == "sex":
                        input = gr.Textbox(
                            label="Gender",
                            value="",
                            info="Gender of the author",
                        )
                    elif attr == "age":
                        input = gr.Textbox(
                            label="Age",
                            value="",
                            info="Age in years, either explicit, e.g. 25, or a range, e.g. 20-30",
                        )
                    elif attr == "occupation":
                        input = gr.Textbox(
                            label="Occupation",
                            value="",
                            info="Brief Occupation Descriptor, e.g. 'Software Engineer'",
                        )
                    elif attr == "birth_city_country":
                        input = gr.Textbox(
                            label="Place of Birth",
                            value="",
                            info="Give the closest city / state / country",
                        )
                    elif attr == "relationship_status":
                        input = gr.Textbox(
                            label="Marital Status",
                            value="",
                            info="Relationship status of the person",
                        )
                    elif attr == "income_level":
                        input = gr.Textbox(
                            label="Income",
                            value="",
                            info="Annual Income Level - No: No Income\nLow: < 30k\nMedium: 30k - 60k\nHigh: 60k - 150k\nVery High: > 150k",
                        )
                    elif attr == "education":
                        input = gr.Textbox(
                            label="Education Level",
                            value="",
                            info="Highest level of education.",
                        )
                    else:
                        raise Exception(f"Unknown attribute {attr}")

                    inputs.append(input)
                    slider = gr.Slider(0, 5, label=f"Hardness", step=1)
                    hardness_sliders.append(slider)
                    slider = gr.Slider(0, 5, label=f"Certainty", step=1)
                    certainty_sliders.append(slider)

        greet_btn = gr.Button("Next")

        with gr.Accordion(label="Full JSON", open=False):
            hidden_box = gr.Textbox(
                label="JSON",
                value="",
                max_lines=2,
            )

        greet_btn.click(
            fn=save_and_load_new,
            inputs=[
                hidden_box,
                *inputs,
                # *dfs_boxes,
                *hardness_sliders,
                *certainty_sliders,
            ],
            outputs=[
                name,
                # anonymized,
                # use_subreddits,
                # expected_labels,
                hidden_box,
                *inputs,
                # *dfs_boxes,
                *hardness_sliders,
                *certainty_sliders,
            ],
        )

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/thread/comments_unique_id.jsonl"
    )
    parser.add_argument("--outpath", type=str, default="data/thread/synth_correct.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user", type=str, default="human") # can use os.getlogin()
    args = parser.parse_args()

    random.seed(args.seed)
    seed = args.seed
    user = args.user
    data = load_data(args.path, args)
    out_path = args.outpath

    if os.path.isfile(".synth_temp"):
        with open(".synth_temp", "r") as f:
            lines = f.readlines()
            old_seed = int(lines[0])
            data_idx = int(lines[1])

        if old_seed != seed:
            data_idx = 7828
    else:
        data_idx = 7828

    pbar = tqdm.tqdm(total=len(data))
    pbar.update(data_idx - 1)
    # Show the bar with new value
    pbar.refresh()

    main(data, args)

    pbar.close()
