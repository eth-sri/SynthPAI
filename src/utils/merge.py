import json
import argparse
from functools import reduce
from sortedcontainers import SortedDict


def merge(a: dict, b: dict, path=[]):

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif isinstance(a[key], list) and isinstance(b[key], list):
                for item in b[key]:
                    if item not in a[key]:
                        a[key].append(item)
            elif a[key] != b[key]:
                if key == "num_comments":
                    a[key] = max(a[key], b[key])
                    print(
                        f" Found conflict at {path} with values {a[key]} and {b[key]} for key {key}"
                    )
                else:
                    raise Exception(
                        f"Conflict at {'.'.join(path + [str(key)])} with values {a[key]} and {b[key]}"
                    )
        else:
            a[key] = b[key]
    return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="Files to merge")
    parser.add_argument("--merge_key", help="Output file")
    parser.add_argument("--outpath", "-o", help="Output file")
    args = parser.parse_args()

    dicts = SortedDict()
    for file in args.files:
        with open(file, "r") as f:
            for i, line in enumerate(f.readlines()):
                data = json.loads(line)
                if args.merge_key is not None:
                    key = data[args.merge_key]
                else:
                    key = i

                if key not in dicts:
                    dicts[key] = []

                dicts[key].append(data)

    merged_dicts = SortedDict()

    for key in dicts.keys():
        if len(dicts[key]) != len(args.files):
            print(key, len(dicts[key]))

    for key, d in dicts.items():
        merged_dicts[key] = reduce(merge, d)

    with open(args.outpath, "w") as f:
        for key in merged_dicts.keys():
            f.write(json.dumps(merged_dicts[key]) + "\n")
