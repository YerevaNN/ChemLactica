import json
import itertools
from transformers import AutoTokenizer, BatchEncoding

jsonl_file_path = "/mnt/sxtn/phil/3953_start.jsonl"
tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m")
CONTEXT_LENGTH = 2048
wrong_count = 0


def get_num_tokens(tokenized):
    return len(tokenized["input_ids"])


def iter_get_num_tokens(list_ass):
    sum = 0
    for a in list_ass:
        sum += get_num_tokens(a)
    return sum


def add_var_str(var_object):
    document = ""
    document += "[VARNAME " + str(var_object["name"]) + "]"
    document += "[VARDESC " + str(var_object["description"]) + "]"
    document += "[VARVALUE " + str(var_object["value"]) + "]"
    return document


def remove_from_all_values(dict_type, num_to_remove):
    for key, value in dict_type.items():
        dict_type[key] = value[:-num_to_remove]
    return dict_type


def evenly_remove_elements_from_lists(lists, total_elements_to_remove):
    # num_lists = len(lists)
    # extra_elements = total_elements_to_remove % num_lists

    # for i in range(num_lists):
    #    num_to_remove = elements_per_list + (1 if i < extra_elements else 0)
    lists[-1] = remove_from_all_values(lists[-1], total_elements_to_remove)
    return lists


def remove_big_assays(assays):
    for assay in assays:
        if len(assay["description"]) > 1400:
            assays.remove(assay)
    return assays


def process_assays(assays):
    assays = json_data["assays"]

    sorted_assays = sorted(assays, key=lambda x: len(x["description"]), reverse=False)

    print("number of assays before removing big ones", len(sorted_assays))
    sorted_assays = remove_big_assays(sorted_assays)
    print("number of assays before removing big ones", len(sorted_assays))
    return sorted_assays


def group_the_lists(list_of_BE_lists):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    number_of_lists = len(list_of_BE_lists)
    for index in range(number_of_lists):
        for element in list_of_BE_lists[index]:
            input_ids.extend(element["input_ids"])
            token_type_ids.extend(element["token_type_ids"])
            attention_mask.extend(element["attention_mask"])

    combined = BatchEncoding(
        {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
    )

    return combined


context_length = CONTEXT_LENGTH
need_new_assay = True
assay_num = 0


with open(jsonl_file_path, "r") as jsonl_file:
    for index, line in enumerate(jsonl_file):
        # Parse the JSON compound from the current line
        json_data = json.loads(json.loads(line))
        sorted_assays = process_assays(json_data["assays"])

        need_new_assay = True
        assay_num = 0
        documents = []

        # Loop until the compound has no more associated assays
        while sorted_assays:
            doc_len_dic = {"desc": 0, "name": 0, "var": 0}
            doc_dict = {"names": [], "descriptions": [], "variables": []}
            tok_ass_vars = []

            # loop until we fill full context
            while (
                doc_len_dic["desc"] + doc_len_dic["name"] + doc_len_dic["var"]
            ) < context_length:
                if need_new_assay:
                    try:
                        assay = sorted_assays.pop()
                        variables = assay["variables"]
                        assay_num += 1
                    except Exception as e:
                        print(e)
                        break
                    tok_ass_name = tokenizer("[ASSNAME " + assay["name"] + "]")
                    tok_ass_desc = tokenizer("[ASSDESC " + assay["description"] + "]")
                if doc_len_dic["name"] == 0 or need_new_assay:
                    doc_dict["names"].append(tok_ass_name)
                    doc_dict["descriptions"].append(tok_ass_desc)
                    doc_len_dic["name"] += get_num_tokens(tok_ass_name)
                    doc_len_dic["desc"] += get_num_tokens(tok_ass_desc)
                    need_new_assay = False
                    continue

                # if current assay has no more data
                if not variables:
                    doc_dict["variables"].append(tok_ass_vars)
                    tok_ass_vars = []
                    need_new_assay = True
                    continue
                else:
                    var_tokens = tokenizer(add_var_str(variables.pop()))
                    doc_len_dic["var"] += get_num_tokens(var_tokens)
                    tok_ass_vars.append(var_tokens)
            if tok_ass_vars:
                doc_dict["variables"].append(tok_ass_vars)

            # check how much to remove from description
            difference = (
                doc_len_dic["desc"] + doc_len_dic["name"] + doc_len_dic["var"]
            ) - context_length
            if difference > 0:
                doc_dict["descriptions"] = evenly_remove_elements_from_lists(
                    doc_dict["descriptions"], difference
                )

            doc_dict["variables"] = list(
                itertools.chain.from_iterable(doc_dict["variables"])
            )

            batch_encoding = group_the_lists(
                [doc_dict["names"], doc_dict["descriptions"], doc_dict["variables"]]
            )
            if get_num_tokens(batch_encoding) == context_length:
                documents.append(batch_encoding)
            else:
                wrong_count += 1

            doc_dict["names"] = []
            doc_dict["descriptions"] = []
            doc_dict["variables"] = []
            tok_ass_vars = []
        break

print(tokenizer.decode(documents[2]["input_ids"]))
print(len(documents[2]["input_ids"]))
print(wrong_count)
print(len(documents))
