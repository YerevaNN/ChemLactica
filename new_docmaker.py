import json
import random


# import itertools
from transformers import AutoTokenizer, BatchEncoding

jsonl_file_path = "/mnt/sxtn/phil/3953_start.jsonl"
tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m")
CONTEXT_LENGTH = 2048
wrong_count = 0
seed_value = 42
random.seed(seed_value)


def get_num_be_tokens(tokenized):
    return len(tokenized["input_ids"])


def add_var_str(var_object):
    # TODO: decide finally assay tags
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
    # removes only the last description
    # TODO: smarter removal of tokens from ends of multiple assay descriptions
    lists[-1] = remove_from_all_values(lists[-1], total_elements_to_remove)
    return lists


def remove_big_assays(assays):
    return [assay for assay in assays if len(assay["description"]) <= 9900]


def process_assays(assays):
    sorted_assays = sorted(assays, key=lambda x: len(x["description"]), reverse=False)
    sorted_assays = remove_big_assays(sorted_assays)
    return sorted_assays


def combine_batch_encoding_lists(list_of_BE_lists, list_of_computed):
    # TODO: pytorch_compatibility
    input_ids = []
    token_type_ids = []
    attention_mask = []

    num_iterations = len(list_of_BE_lists[0])
    for i in range(num_iterations):
        for index, interest_list in enumerate(list_of_BE_lists):
            if index == 2:
                try:
                    sub_var_list = interest_list[i]
                    for actual_var in sub_var_list:
                        input_ids.extend(actual_var["input_ids"])
                        token_type_ids.extend(actual_var["token_type_ids"])
                        attention_mask.extend(actual_var["attention_mask"])
                except IndexError:
                    pass
            else:
                input_ids.extend(interest_list[i]["input_ids"])
                token_type_ids.extend(interest_list[i]["token_type_ids"])
                attention_mask.extend(interest_list[i]["attention_mask"])
    for comp_prop in list_of_computed:
        input_ids.extend(comp_prop["input_ids"])
        token_type_ids.extend(comp_prop["token_type_ids"])
        attention_mask.extend(comp_prop["attention_mask"])

    combined = BatchEncoding(
        {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
    )

    return combined


def create_assay_base(tokenizer, assay):
    tok_ass_name = tokenizer("[ASSNAME " + assay["name"] + "]")
    tok_ass_desc = tokenizer("[ASSDESC " + assay["description"] + "]")
    return tok_ass_name, tok_ass_desc


def extract_data_from_json(json_data):
    sorted_assays = process_assays(json_data["assays"])

    # smiles = json_data["SMILES"]
    computed_dict = {}
    for key, value in json_data.items():
        if isinstance(value, list) or value == "SMILES":
            continue
        comp_val = tokenizer(f"[{str(key).upper()}" + str(value) + "]")
        comp_len = get_num_be_tokens(comp_val)
        computed_dict[comp_len] = comp_val
    return sorted_assays, computed_dict


context_length = CONTEXT_LENGTH
need_new_assay = True


with open(jsonl_file_path, "r") as jsonl_file:
    for index, line in enumerate(jsonl_file):
        json_data = json.loads(json.loads(line))

        # Parse the compound associated data from the current line
        sorted_assays, computed_dict = extract_data_from_json(json_data)
        need_new_assay = True
        documents = []

        # Loop until the compound has no more associated assays
        while sorted_assays:
            doc_len = 0
            document_content_dict = {
                "names": [],
                "descriptions": [],
                "variables": [],
                "computed": [],
            }
            tok_ass_vars = []

            # loop until we fill full context
            while (doc_len) < context_length:
                if need_new_assay:
                    try:
                        assay = sorted_assays.pop()
                        tok_ass_name, tok_ass_desc = create_assay_base(tokenizer, assay)
                        variables = assay["variables"]
                    except Exception as e:
                        print(e)
                        break

                if doc_len == 0 or need_new_assay:
                    ass_name_len = get_num_be_tokens(tok_ass_name)
                    ass_desc_len = get_num_be_tokens(tok_ass_desc)

                    if computed_dict and not doc_len == 0:
                        potential_next_length = ass_name_len + ass_desc_len
                        if (ass_name_len + ass_desc_len + doc_len) > context_length:
                            diff = context_length - (doc_len)
                            while diff > 0:
                                try:
                                    random_key = random.choice(
                                        list(computed_dict.keys())
                                    )
                                    value = computed_dict.pop(random_key)
                                    document_content_dict["computed"].append(value)
                                    doc_len += random_key
                                    diff -= random_key
                                except Exception:
                                    break
                            continue
                    document_content_dict["names"].append(tok_ass_name)
                    document_content_dict["descriptions"].append(tok_ass_desc)
                    doc_len += ass_name_len
                    doc_len += ass_desc_len
                    need_new_assay = False
                    continue

                # if current assay has no more data
                if not variables:
                    document_content_dict["variables"].append(tok_ass_vars)
                    tok_ass_vars = []
                    need_new_assay = True
                    continue
                # if it has data, add it
                else:
                    var_tokens = tokenizer(add_var_str(variables.pop()))
                    doc_len += get_num_be_tokens(var_tokens)
                    tok_ass_vars.append(var_tokens)

            if tok_ass_vars:
                document_content_dict["variables"].append(tok_ass_vars)

            # check how many tokens to remove from description
            difference = (doc_len) - context_length

            if difference > 0:
                document_content_dict[
                    "descriptions"
                ] = evenly_remove_elements_from_lists(
                    document_content_dict["descriptions"], difference
                )

            doc_batch_encoding = combine_batch_encoding_lists(
                [
                    document_content_dict["names"],
                    document_content_dict["descriptions"],
                    document_content_dict["variables"],
                ],
                document_content_dict["computed"],
            )

            if get_num_be_tokens(doc_batch_encoding) == context_length:
                documents.append(doc_batch_encoding)
            else:
                print("num tokens here", get_num_be_tokens(doc_batch_encoding))
                wrong_count += 1
        break
# print(tokenizer.decode(documents[2]["input_ids"]))
print("num docs", len(documents))
print("wrong count:", wrong_count)
