import random
import torch


def get_num_be_tokens(tokenized):
    return len(tokenized["input_ids"])


def add_var_str(var_object):
    var_text = f"""[ASSAY_VAR {var_object['name']} DESC {var_object['description']} VAL {var_object['value']}]"""  # noqa
    return var_text


def remove_from_all_values(dict_type, num_to_remove):
    for key, value in dict_type.items():
        dict_type[key] = value[: -(num_to_remove + 1)] + [value[-1]]
    return dict_type


def evenly_remove_elements_from_lists(lists, total_elements_to_remove):
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


def combine_batch_encodings(document_content_dict, doc_start):
    # TODO: pytorch_compatibility
    input_ids = torch.empty(0, dtype=torch.int64)
    token_type_ids = torch.empty(0, dtype=torch.int64)
    attention_mask = torch.empty(0, dtype=torch.int64)
    input_ids = torch.cat((input_ids, torch.tensor(doc_start["input_ids"])))
    token_type_ids = torch.cat(
        (token_type_ids, torch.tensor(doc_start["token_type_ids"]))
    )
    attention_mask = torch.cat(
        (attention_mask, torch.tensor(doc_start["attention_mask"]))
    )

    for index, element in enumerate(document_content_dict["computed"]):
        if element["name"] == "SMILES":
            smiles_index = index
    if random.random() < 0.5:
        smiles_prop = document_content_dict["computed"].pop(smiles_index)
        input_ids = torch.cat(
            (
                input_ids,
                torch.tensor(smiles_prop["value"]["input_ids"], dtype=torch.int64),
            )
        )
        token_type_ids = torch.cat(
            (
                token_type_ids,
                torch.tensor(smiles_prop["value"]["token_type_ids"], dtype=torch.int64),
            )
        )
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.tensor(smiles_prop["value"]["attention_mask"], dtype=torch.int64),
            )
        )

    num_iterations = len(document_content_dict["names"])
    for i in range(num_iterations):
        for key, interest_list in document_content_dict.items():
            if key == "variables":
                try:
                    sub_var_list = interest_list[i]
                    for actual_var in sub_var_list:
                        input_ids = torch.cat(
                            (
                                input_ids,
                                torch.tensor(
                                    actual_var["input_ids"], dtype=torch.int64
                                ),
                            )
                        )
                        token_type_ids = torch.cat(
                            (
                                token_type_ids,
                                torch.tensor(
                                    actual_var["token_type_ids"], dtype=torch.int64
                                ),
                            )
                        )
                        attention_mask = torch.cat(
                            (
                                attention_mask,
                                torch.tensor(
                                    actual_var["attention_mask"], dtype=torch.int64
                                ),
                            )
                        )
                except IndexError:
                    pass
            elif key == "computed":
                continue
            else:
                input_ids = torch.cat(
                    (
                        input_ids,
                        torch.tensor(interest_list[i]["input_ids"], dtype=torch.int64),
                    )
                )
                token_type_ids = torch.cat(
                    (
                        token_type_ids,
                        torch.tensor(
                            interest_list[i]["token_type_ids"], dtype=torch.int64
                        ),
                    )
                )
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.tensor(
                            interest_list[i]["attention_mask"], dtype=torch.int64
                        ),
                    )
                )

    for comp_prop in document_content_dict["computed"]:
        input_ids = torch.cat(
            (
                input_ids,
                torch.tensor(comp_prop["value"]["input_ids"], dtype=torch.int64),
            )
        )
        token_type_ids = torch.cat(
            (
                token_type_ids,
                torch.tensor(comp_prop["value"]["token_type_ids"], dtype=torch.int64),
            )
        )
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.tensor(comp_prop["value"]["attention_mask"], dtype=torch.int64),
            )
        )

    return input_ids[:2048], token_type_ids[:2048], attention_mask[:2048]


def create_assay_base(tokenizer, assay):
    tok_ass_name = tokenizer(f"""[ASSAY_NAME {str(assay["name"])}]""")
    tok_ass_desc = tokenizer(f"""[ASSAY_DESC {str(assay["description"])}]""")
    return tok_ass_name, tok_ass_desc


def extract_data_from_json(json_data, tokenizer):
    sorted_assays = process_assays(json_data["assays"])

    computed_dict = {
        "synonyms": [],
        "related": [],
        "experimental": [],
    }
    related_count = 0
    for key, value in json_data.items():
        if key == "SMILES":
            continue
        if key == "related":
            for list_val in value:
                related_count += 1
                comp_val = tokenizer(
                    f"""[SIMILARITY {str(list_val["similarity"])} SMILES {list_val["SMILES"]}]"""
                )
                computed_dict[key].append(comp_val)
            continue
        if key == "synonyms":
            for list_val in value:
                comp_val = tokenizer(f"""[SYNONYM {list_val["name"]}]""")
                computed_dict[key].append(comp_val)
            continue

        if key == "experimental":
            for list_val in value:
                comp_val = tokenizer(
                    f"""[EXPERIMENTAL {list_val["PROPERTY_NAME"]} {list_val["PROPERTY_VALUE"]}]"""
                )
                computed_dict[key].append(comp_val)
            continue
        else:
            comp_val = tokenizer(f"""[{str(key).upper()} {str(value)}]""")
            computed_dict[key] = comp_val
    return sorted_assays, computed_dict


def get_compound_assay_docs(tokenizer, json_data, context_length=2048):
    need_new_assay = True
    # Parse the compound associated data from the current line
    sorted_assays, computed_dict = extract_data_from_json(json_data, tokenizer)
    smiles = "[START_SMILES]" + json_data["SMILES"] + "[END_SMILES]"
    smiles_toks = tokenizer(smiles)
    doc_start = tokenizer("</s>")
    need_new_assay = True
    documents = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
    }
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

        # loop until we fill full context
        doc_len += get_num_be_tokens(doc_start)
        while (doc_len) < context_length:
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
                    if (ass_name_len + ass_desc_len + doc_len) > context_length:
                        diff = context_length - (doc_len)
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
        difference = (doc_len) - context_length

        if difference > 0:
            try:
                document_content_dict[
                    "descriptions"
                ] = evenly_remove_elements_from_lists(
                    document_content_dict["descriptions"], difference
                )
            except Exception:
                pass

        doc_input_ids, doc_token_type_ids, doc_attention_mask = combine_batch_encodings(
            document_content_dict, doc_start
        )

        if len(doc_input_ids) == context_length:
            documents["input_ids"].append(doc_input_ids)
            documents["token_type_ids"].append(doc_token_type_ids)
            documents["attention_mask"].append(doc_attention_mask)
        else:
            wrong_count += 1
    return documents
