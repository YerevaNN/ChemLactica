import random

# import traceback
# import pickle
import json
import argparse
import time
import torch
from transformers import BatchEncoding, AutoTokenizer


def modified_tokenizer_call(tokenizer, text):
    be = tokenizer(text, return_tensors="pt", return_token_type_ids=False)
    for key, be_tensor in be.items():
        be[key] = be_tensor.squeeze()
        if be[key].dim() == 0:
            be[key] = be[key].unsqueeze(0)
    return be


def get_num_be_tokens(tokenized):
    result = len(tokenized["input_ids"])
    return result


def create_var_sub_tag(var_object, string_key, open_tag, close_tag):
    if var_object[string_key] != "":
        tagged_component = f"{open_tag}{var_object[string_key]}{close_tag}"
    else:
        tagged_component = ""
    return tagged_component


def add_var_str(var_object):
    name_tag_component = create_var_sub_tag(
        var_object, "name", "[VAR_NAME]", "[/VAR_NAME]"
    )
    desc_tag_component = create_var_sub_tag(
        var_object, "description", "[VAR_DESC]", "[/VAR_DESC]"
    )
    val_tag_component = create_var_sub_tag(
        var_object, "value", "[VAR_VAL]", "[/VAR_VAL]"
    )
    unit_tag_component = create_var_sub_tag(
        var_object, "unit", "[VAR_UNIT]", "[/VAR_UNIT]"
    )
    var_text = f"""{name_tag_component}{desc_tag_component}{unit_tag_component}{val_tag_component}"""  # noqa
    return var_text


def remove_from_all_values(dict_type, num_to_remove):
    num_to_remove = num_to_remove + 1
    for key, value in dict_type.items():
        dict_type[key] = torch.cat(
            (value[:-num_to_remove], value[-1].unsqueeze(0)), dim=0
        )
        # dict_type[key] = value[: -(num_to_remove + 1)] + [value[-1]]
    return dict_type


def evenly_remove_elements_from_lists(lists, total_elements_to_remove):
    lists[-1] = remove_from_all_values(lists[-1], total_elements_to_remove)
    return lists


def sp_remove_from_all_values(dict_type, num_to_remove):
    num_to_remove = num_to_remove + 1
    for key, value in dict_type.items():
        dict_type[key] = torch.cat(
            (value[:-num_to_remove], value[-1].unsqueeze(0)), dim=0
        )
        # dict_type[key] = value[: -(num_to_remove + 1)] + [value[-1]]
    return dict_type


def sp_evenly_remove_elements_from_lists(lists, total_elements_to_remove):
    lists[-1] = sp_remove_from_all_values(lists[-1], total_elements_to_remove)
    return lists


def remove_big_assays(assays):
    return [assay for assay in assays if len(assay["description"]) <= 2500]


def process_assays(assays):
    sorted_assays = sorted(assays, key=lambda x: len(x["description"]), reverse=False)
    sorted_assays = remove_big_assays(sorted_assays)
    return sorted_assays


def extend_be(base_be, new_be):
    for key, be_tensor in new_be.items():
        base_be[key] = torch.cat((base_be[key], be_tensor))
    return base_be


def combine_batch_encodings(document_content_dict, doc_start, model_context_length):
    doc_be = BatchEncoding(
        {
            "input_ids": torch.empty(0, dtype=torch.int64),
            # "token_type_ids": torch.empty(0, dtype=torch.int64),
            "attention_mask": torch.empty(0, dtype=torch.int64),
        }
    )
    doc_be = extend_be(doc_be, doc_start)

    for index, element in enumerate(document_content_dict["computed"]):
        if element["name"] == "SMILES":
            smiles_index = index
    if random.random() < 0.5:
        smiles_prop = document_content_dict["computed"].pop(smiles_index)
        doc_be = extend_be(doc_be, smiles_prop["value"])

    num_iterations = len(document_content_dict["names"])
    for i in range(num_iterations):
        for key, interest_list in document_content_dict.items():
            if key == "variables":
                try:
                    sub_var_list = interest_list[i]
                    for actual_var in sub_var_list:
                        doc_be = extend_be(doc_be, actual_var)
                except IndexError:
                    pass
            elif key == "computed":
                continue
            else:
                doc_be = extend_be(doc_be, interest_list[i])

    for comp_prop in document_content_dict["computed"]:
        if comp_prop["name"] == "SMILES" and (
            get_num_be_tokens(comp_prop["value"]) + get_num_be_tokens(doc_be)
            > model_context_length
        ):
            final_diff = (
                get_num_be_tokens(doc_be)
                + get_num_be_tokens(comp_prop["value"])
                - model_context_length
            )
            doc_be["input_ids"] = doc_be["input_ids"][:-final_diff]
            # doc_be["token_type_ids"] = doc_be["token_type_ids"][:-final_diff]
            doc_be["attention_mask"] = doc_be["attention_mask"][:-final_diff]
        doc_be = extend_be(doc_be, comp_prop["value"])

    input_ids = doc_be["input_ids"]
    # token_type_ids = doc_be["token_type_ids"]
    attention_mask = doc_be["attention_mask"]

    return (
        input_ids[:model_context_length],
        # token_type_ids[:model_context_length],
        attention_mask[:model_context_length],
    )


def create_assay_base(tokenizer, assay):
    tok_ass_name = modified_tokenizer_call(
        tokenizer, f"""[ASSAY_NAME]{str(assay["name"])}[/ASSAY_NAME]"""
    )
    tok_ass_desc = modified_tokenizer_call(
        tokenizer, f"""[ASSAY_DESC]{str(assay["description"])}[/ASSAY_DESC]"""
    )
    return tok_ass_name, tok_ass_desc


def get_computed_dict(json_data, tokenizer):
    computed_dict = {
        "synonyms": [],
        "related": [],
        "experimental": [],
    }

    for key, value in json_data.items():
        if key == "SMILES" or key == "assays":
            continue
        if key == "related":
            for list_val in value:
                comp_val = modified_tokenizer_call(
                    tokenizer,
                    f"""[SIMILAR]{str(list_val["similarity"])} {list_val["SMILES"]}[/SIMILAR]""",  # noqa
                )
                computed_dict[key].append(comp_val)
            continue
        if key == "synonyms":
            for list_val in value:
                comp_val = modified_tokenizer_call(
                    tokenizer, f"""[SYNONYM]{list_val["name"]}[/SYNONYM]"""
                )
                computed_dict[key].append(comp_val)
            continue

        if key == "experimental":
            for list_val in value:
                comp_val = modified_tokenizer_call(
                    tokenizer,
                    f"""[PROPERTY]{list_val["PROPERTY_NAME"]} {list_val["PROPERTY_VALUE"]}[/PROPERTY]""",  # noqa
                )
                computed_dict[key].append(comp_val)
            continue
        else:
            comp_val = modified_tokenizer_call(
                tokenizer, f"""[{str(key).upper()}]{str(value)}[/{str(key).upper()}]"""
            )
            computed_dict[key] = comp_val
    return computed_dict


def extract_data_from_json(json_data, tokenizer):
    sorted_assays = process_assays(json_data["assays"])
    computed_dict = get_computed_dict(json_data, tokenizer)

    return sorted_assays, computed_dict


def get_compound_assay_docs(tokenizer, json_data, model_context_length):
    need_new_assay = True
    # Parse the compound associated data from the current line
    sorted_assays, computed_dict = extract_data_from_json(json_data, tokenizer)
    smiles = "[START_SMILES]" + json_data["SMILES"] + "[END_SMILES]"
    smiles_toks = modified_tokenizer_call(tokenizer, smiles)
    doc_start = modified_tokenizer_call(tokenizer, "</s>")
    documents = {
        "input_ids": [],
        # "token_type_ids": [],
        "attention_mask": [],
    }
    # wrong_count = 0

    # Loop until the compound has no more associated assays
    while sorted_assays:
        doc_len = 0
        document_content_dict = {
            "names": [],
            "descriptions": [],
            "variables": [],
            "computed": [],
        }
        incomplete_doc = None
        tok_ass_vars = []
        # document_content_dict["computed"].append({"name": "SMILES","value":smiles_toks})
        # doc_len += get_num_be_tokens(smiles_toks)

        # loop until we fill full context
        doc_len += get_num_be_tokens(doc_start)
        while (doc_len) < model_context_length:
            if doc_len == get_num_be_tokens(doc_start):
                document_content_dict["computed"].append(
                    {"name": "SMILES", "value": smiles_toks}
                )
                doc_len += get_num_be_tokens(smiles_toks)
                continue
            if need_new_assay:
                try:
                    assay = sorted_assays.pop()
                    tok_ass_name, tok_ass_desc = create_assay_base(tokenizer, assay)
                    variables = assay["variables"]
                except IndexError:
                    break

            if (
                doc_len == get_num_be_tokens(smiles_toks) + get_num_be_tokens(doc_start)
                or need_new_assay
            ):
                ass_name_len = get_num_be_tokens(tok_ass_name)
                ass_desc_len = get_num_be_tokens(tok_ass_desc)

                if computed_dict and not doc_len == get_num_be_tokens(
                    doc_start
                ) + get_num_be_tokens(smiles_toks):
                    if (ass_name_len + ass_desc_len + doc_len) > model_context_length:
                        diff = model_context_length - (doc_len)
                        while diff > 0:
                            try:
                                random_key = random.choice(list(computed_dict.keys()))
                                if random_key in [
                                    "synonyms",
                                    "related",
                                    "experimental",
                                ]:
                                    if not computed_dict[random_key]:
                                        del computed_dict[random_key]
                                    else:
                                        value = computed_dict[random_key].pop()
                                else:
                                    value = computed_dict.pop(random_key)

                                document_content_dict["computed"].append(
                                    {"name": random_key, "value": value}
                                )

                                doc_len += get_num_be_tokens(value)
                                diff -= get_num_be_tokens(value)
                            except (IndexError, UnboundLocalError):
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
                var_tokens = modified_tokenizer_call(
                    tokenizer, add_var_str(variables.pop())
                )
                doc_len += get_num_be_tokens(var_tokens)
                tok_ass_vars.append(var_tokens)

        if tok_ass_vars:
            document_content_dict["variables"].append(tok_ass_vars)

        # check how many tokens to remove from description
        difference = (doc_len) - model_context_length

        inc = False

        if difference == 0:
            inc = False
            pass
        elif difference < 0:
            inc = True
            incomplete_doc = {
                "doc_dic": document_content_dict,
                "doc_len": doc_len,
                "doc_smiles": smiles,
            }

        if difference > 0:
            inc = False
            document_content_dict["descriptions"] = evenly_remove_elements_from_lists(
                document_content_dict["descriptions"], difference
            )
        if not inc:
            (
                doc_input_ids,
                # doc_token_type_ids,
                doc_attention_mask,
            ) = combine_batch_encodings(
                document_content_dict, doc_start, model_context_length
            )

            if len(doc_input_ids) == model_context_length:
                documents["input_ids"].append(doc_input_ids)
                # documents["token_type_ids"].append(doc_token_type_ids)
                documents["attention_mask"].append(doc_attention_mask)
    return documents, incomplete_doc


def process_incomplete_docs(incomplete_docs, tokenizer, model_context_length):
    doc_start = modified_tokenizer_call(tokenizer, "</s>")
    documents = {
        "input_ids": [],
        # "token_type_ids": [],
        "attention_mask": [],
    }
    while incomplete_docs:
        new_doc_len = 0
        doc_be = BatchEncoding(
            {
                "input_ids": torch.empty(0, dtype=torch.int64),
                # "token_type_ids": torch.empty(0, dtype=torch.int64),
                "attention_mask": torch.empty(0, dtype=torch.int64),
            }
        )
        while True:
            try:
                incomplete_doc = incomplete_docs.pop()
                new_doc_len += incomplete_doc["doc_len"]
            except IndexError:
                break
            if new_doc_len < model_context_length:
                input_ids, attention_mask = combine_batch_encodings(
                    incomplete_doc["doc_dic"], doc_start, model_context_length
                )
                doc_be = extend_be(
                    doc_be,
                    BatchEncoding(
                        {
                            "input_ids": input_ids,
                            # "token_type_ids": token_type_ids,
                            "attention_mask": attention_mask,
                        }
                    ),
                )  # noqa
            elif new_doc_len >= model_context_length:
                difference = new_doc_len - model_context_length
                incomplete_doc["doc_dic"][
                    "descriptions"
                ] = sp_evenly_remove_elements_from_lists(
                    incomplete_doc["doc_dic"]["descriptions"], difference
                )
                input_ids, attention_mask = combine_batch_encodings(
                    incomplete_doc["doc_dic"], doc_start, model_context_length
                )
                len(input_ids)
                doc_be = extend_be(
                    doc_be,
                    BatchEncoding(
                        {
                            "input_ids": input_ids,
                            # "token_type_ids": token_type_ids,
                            "attention_mask": attention_mask,
                        }
                    ),
                )  # noqa
                break

        if len(doc_be["input_ids"]) > model_context_length:
            documents["input_ids"].append(doc_be["input_ids"][:model_context_length])
            # documents["token_type_ids"].append(
            #     doc_be["token_type_ids"][:model_context_length]
            # )
            documents["attention_mask"].append(
                doc_be["attention_mask"][:model_context_length]
            )
        # print(tokenizer.decode(doc_be["input_ids"]))
        # print("doc made")
    return documents


def main(jsonl_file_path, tokenizer_id):
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    GALACTICA_CONTEXT_LENGTH = 2048
    seed_value = 42
    # wrong_count = 0
    random.seed(seed_value)
    # numbers_of_docs = []
    incomplete_docs = []

    with open(jsonl_file_path, "r") as jsonl_file:
        for index, line in enumerate(jsonl_file):
            if index < 9:
                continue
            print(index)
            json_data = json.loads(json.loads(line))
            # try:
            documents, incomplete_doc = get_compound_assay_docs(
                tokenizer, json_data, GALACTICA_CONTEXT_LENGTH
            )
            if incomplete_doc:
                incomplete_docs.append(incomplete_doc)
            # except Exception as e:
            # print(e)
            # continue
            # print("num docs", len(documents["input_ids"]))
            # numbers_of_docs.append(len(documents["input_ids"]))
            if index > 10:
                break
        # with open("numbers_of_docs.pkl", "wb") as file:
        #     pickle.dump(numbers_of_docs, file)
        end = time.time()
        diff = end - start
        print("time elapsed", diff)
        fixed_docs = process_incomplete_docs(incomplete_docs, tokenizer)
        print(len(fixed_docs["input_ids"]))
        # print(len(fixed_docs["input_ids"][0]))
        print(tokenizer.decode(fixed_docs["input_ids"][0]))
        # print("---------------------------")
        # print(tokenizer.decode(documents["input_ids"][6]))
        # print("----------------------------")
        # print(tokenizer.decode(documents["input_ids"][8]))
        # print("wrong count:", wrong_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("new doc maker test")
    parser.add_argument("--jsonl_file_path", type=str, help="Path to the JSONL file")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer name or configuration")
    args = parser.parse_args()
    main(args.jsonl_file_path, args.tokenizer)
