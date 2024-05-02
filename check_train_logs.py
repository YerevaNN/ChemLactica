import argparse
import json
import re
import glob


def extract_text_between_index_tags(text):
    pattern = r"\[INDEX\](.*?)\[/INDEX\]"
    matches = re.findall(pattern, text)
    return matches


def main(args):
    num_indices_expected = args.max_index + 1
    index_counter = {key: 0 for key in range(num_indices_expected)}
    train_file_counter = {}
    yield_file_counter = {}
    log_files = glob.glob(args.log_path + "/*train.txt")
    yield_files = glob.glob(args.log_path + "/*yield.txt")
    print("files", log_files)

    for log_file in log_files:
        train_file_counter[log_file] = 0
        with open(log_file, "r") as f:
            for line in f.readlines():
                line_index_values = extract_text_between_index_tags(line)
                for line_index_value in line_index_values:
                    index_counter[int(line_index_value)] += 1
                    train_file_counter[log_file] += 1
    for yield_file in yield_files:
        yield_file_counter[yield_file] = 0
        with open(yield_file, "r") as yf:
            for line in yf.readlines():
                index = json.loads(line)["index"]
                if index is not None:
                    yield_file_counter[yield_file] += 1

    total_num_indices_seen = sum(index_counter.values())

    with open(args.analysis_output, "w") as f:
        for yield_log_file, num_indices in yield_file_counter.items():
            f.write(f"{yield_log_file}, {num_indices}\n")

        for train_log_file, num_indices in train_file_counter.items():
            f.write(f"{train_log_file}, {num_indices}\n")

        f.write(
            f"total_num_indices_seen: {total_num_indices_seen} and num_indices_expected: {num_indices_expected}\n"  # noqa
        )
        f.write(
            "difference: " + str(total_num_indices_seen - num_indices_expected) + "\n"
        )
        f.write("-----------erroneous index, num_occurences---------------\n")
        for key, stored_index in index_counter.items():
            if stored_index != 1:
                f.write(f"index: {key}, num_occurences: {stored_index}" + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="none")
    parser.add_argument("--log_path", type=str, dest="log_path", required=True)
    parser.add_argument("--output", type=str, dest="analysis_output", required=True)
    parser.add_argument("--max_index", type=int, dest="max_index", required=True)
    args = parser.parse_args()
    main(args)
