import argparse
import re
import glob


def extract_text_between_index_tags(text):
    pattern = r"\[INDEX\](.*?)\[/INDEX\]"
    matches = re.findall(pattern, text)
    return matches


def main(args):
    index_counter = {key: 0 for key in range(args.max_index + 1)}
    log_files = glob.glob(args.log_path + "/*train.txt")
    print("files", log_files)
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f.readlines():
                line_index_values = extract_text_between_index_tags(line)
                for line_index_value in line_index_values:
                    index_counter[int(line_index_value)] += 1

    for key, value in index_counter.items():
        if value != 1:
            print(f"index: {key}, num_occurences: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="none")
    parser.add_argument("--log_path", type=str, dest="log_path", required=True)
    parser.add_argument("--max_index", type=int, dest="max_index", required=True)
    args = parser.parse_args()
    main(args)
