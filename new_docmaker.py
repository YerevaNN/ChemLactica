import json
import argparse
import random
from transformers import AutoTokenizer, BatchEncoding


def get_num_be_tokens(tokenized):
    return len(tokenized["input_ids"])


def add_var_str(var_object):
    var_text = f"""[VARNAME {var_object['name']}][VARDESC {var_object['description']}][VARVALUE {var_object['value']}]"""  # noqa
    return var_text


def remove_from_all_values(dict_type, num_to_remove):
    for key, value in dict_type.items():
        dict_type[key] = value[: -(num_to_remove + 1)] + [value[-1]]
    return dict_type


def evenly_remove_elements_from_lists(lists, total_elements_to_remove):
    # removes only the last description
    # TODO: smarter removal of tokens from ends of multiple assay descriptions
    # print(get_num_be_tokens(lists[-1]))
    # print("elements to remove", total_elements_to_remove)
    lists[-1] = remove_from_all_values(lists[-1], total_elements_to_remove)
    # print(type(lists[-1]))
    # print(get_num_be_tokens(lists[-1]))
    # assert 1==0
    return lists


def remove_big_assays(assays):
    return [assay for assay in assays if len(assay["description"]) <= 2500]


def process_assays(assays):
    sorted_assays = sorted(assays, key=lambda x: len(x["description"]), reverse=False)
    sorted_assays = remove_big_assays(sorted_assays)
    return sorted_assays


def combine_batch_encodings(document_content_dict):
    # TODO: pytorch_compatibility
    input_ids = []
    token_type_ids = []
    attention_mask = []

    for index, element in enumerate(document_content_dict["computed"]):
        if element["name"] == "SMILES":
            smiles_index = index
    if random.random() < 0.5:
        smiles_prop = document_content_dict["computed"].pop(smiles_index)
        input_ids.extend(smiles_prop["value"]["input_ids"])
        token_type_ids.extend(smiles_prop["value"]["token_type_ids"])
        attention_mask.extend(smiles_prop["value"]["attention_mask"])

    num_iterations = len(document_content_dict["names"])
    for i in range(num_iterations):
        for key, interest_list in document_content_dict.items():
            if key == "variables":
                try:
                    sub_var_list = interest_list[i]
                    for actual_var in sub_var_list:
                        input_ids.extend(actual_var["input_ids"])
                        token_type_ids.extend(actual_var["token_type_ids"])
                        attention_mask.extend(actual_var["attention_mask"])
                except IndexError:
                    pass
            elif key == "computed":
                continue
            else:
                input_ids.extend(interest_list[i]["input_ids"])
                token_type_ids.extend(interest_list[i]["token_type_ids"])
                attention_mask.extend(interest_list[i]["attention_mask"])
    for comp_prop in document_content_dict["computed"]:
        input_ids.extend(comp_prop["value"]["input_ids"])
        token_type_ids.extend(comp_prop["value"]["token_type_ids"])
        attention_mask.extend(comp_prop["value"]["attention_mask"])

    combined = BatchEncoding(
        {
            "input_ids": input_ids[:2048],
            "token_type_ids": token_type_ids[:2048],
            "attention_mask": attention_mask[:2048],
        }
    )

    return combined


def create_assay_base(tokenizer, assay):
    tok_ass_name = tokenizer(f"""[ASSNAME {str(assay["name"])}]""")
    tok_ass_desc = tokenizer(f"""[ASSDESC {str(assay["description"])}]""")
    return tok_ass_name, tok_ass_desc


def extract_data_from_json(json_data, tokenizer):
    sorted_assays = process_assays(json_data["assays"])

    computed_dict = {}
    for key, value in json_data.items():
        if isinstance(value, list) or key == "SMILES":
            continue
        comp_val = tokenizer(f"""[{str(key).upper()} {str(value)}]""")
        computed_dict[key] = comp_val
    return sorted_assays, computed_dict


def main(jsonl_file_path, tokenizer_id):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    GALACTICA_CONTEXT_LENGTH = 2048
    wrong_count = 0
    seed_value = 42
    random.seed(seed_value)
    need_new_assay = True

    with open(jsonl_file_path, "r") as jsonl_file:
        for index, line in enumerate(jsonl_file):
            print(index)
            json_data = json.loads(json.loads(line))

            # Parse the compound associated data from the current line
            sorted_assays, computed_dict = extract_data_from_json(json_data, tokenizer)
            smiles = "[START_SMILES]" + json_data["SMILES"] + "[END_SMILES]"
            smiles_toks = tokenizer(smiles)
            need_new_assay = True
            documents = []
            doc_num = 0
            wrong_count = 0

            # Loop until the compound has no more associated assays
            while sorted_assays:
                doc_num += 1
                doc_len = 0
                document_content_dict = {
                    "names": [],
                    "descriptions": [],
                    "variables": [],
                    "computed": [],
                }
                tok_ass_vars = []
                # document_content_dict["computed"].append({"name": "SMILES","value":smiles_toks})
                # doc_len += get_num_be_tokens(smiles_toks)
                needs_smiles = True

                # loop until we fill full context
                while (doc_len) < GALACTICA_CONTEXT_LENGTH:
                    if needs_smiles:
                        document_content_dict["computed"].append(
                            {"name": "SMILES", "value": smiles_toks}
                        )
                        doc_len += get_num_be_tokens(smiles_toks)
                        needs_smiles = False
                        continue
                    if need_new_assay:
                        try:
                            assay = sorted_assays.pop()
                            tok_ass_name, tok_ass_desc = create_assay_base(
                                tokenizer, assay
                            )
                            variables = assay["variables"]
                        except IndexError:
                            break

                    if doc_len == get_num_be_tokens(smiles_toks) or need_new_assay:
                        ass_name_len = get_num_be_tokens(tok_ass_name)
                        ass_desc_len = get_num_be_tokens(tok_ass_desc)

                        if computed_dict and not doc_len == 0:
                            if (
                                ass_name_len + ass_desc_len + doc_len
                            ) > GALACTICA_CONTEXT_LENGTH:
                                diff = GALACTICA_CONTEXT_LENGTH - (doc_len)
                                while diff > 0:
                                    try:
                                        random_key = random.choice(
                                            list(computed_dict.keys())
                                        )
                                        value = computed_dict.pop(random_key)
                                        document_content_dict["computed"].append(
                                            {"name": random_key, "value": value}
                                        )
                                        doc_len += get_num_be_tokens(value)
                                        diff -= get_num_be_tokens(value)
                                    except IndexError:
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
                difference = (doc_len) - GALACTICA_CONTEXT_LENGTH

                if difference > 0:
                    try:
                        document_content_dict[
                            "descriptions"
                        ] = evenly_remove_elements_from_lists(
                            document_content_dict["descriptions"], difference
                        )
                    except Exception:
                        pass

                doc_batch_encoding = combine_batch_encodings(document_content_dict)

                if get_num_be_tokens(doc_batch_encoding) == GALACTICA_CONTEXT_LENGTH:
                    documents.append(doc_batch_encoding)
                else:
                    if len(document_content_dict["descriptions"]) == 1:
                        # print(tokenizer.decode(doc_batch_encoding["input_ids"]))
                        print(len(document_content_dict["descriptions"]))
                        # assert 1==0

                    print("num tokens here", get_num_be_tokens(doc_batch_encoding))
                    wrong_count += 1

            # print(tokenizer.decode(documents[2]["input_ids"]))
            print("num docs", len(documents))
            print("wrong count:", wrong_count)
            if index > 10:
                break
        print(tokenizer.decode(documents[0]["input_ids"]))
        print("---------------------------")
        print(tokenizer.decode(documents[1]["input_ids"]))
        print("----------------------------")
        print(tokenizer.decode(documents[2]["input_ids"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("new doc maker test")
    parser.add_argument("--jsonl_file_path", type=str, help="Path to the JSONL file")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer name or configuration")
    args = parser.parse_args()
    main(args.jsonl_file_path, args.tokenizer)
