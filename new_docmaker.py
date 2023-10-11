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
    document += "[VARNAME" + str(var_object["name"]) + "]"
    document += "[VARDESC" + str(var_object["description"]) + "]"
    document += "[VARVALUE" + str(var_object["value"]) + "]"
    return document


def remove_from_all_values(dict_type, num_to_remove):
    for key, value in dict_type.items():
        dict_type[key] = value[:-num_to_remove]
    return dict_type


def evenly_remove_elements_from_lists(lists, total_elements_to_remove):
    num_lists = len(lists)
    print("number of lists", num_lists)
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


def group_the_lists(list_of_BE_lists):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    number_of_lists = len(list_of_BE_lists)
    for index in range(number_of_lists):
        for element in list_of_BE_lists[index]:
            # print("element",type(element))
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


with open(jsonl_file_path, "r") as jsonl_file:
    for index, line in enumerate(jsonl_file):
        # Parse the JSON object from the current line
        json_data = json.loads(json.loads(line))
        documents = []

        assays = json_data["assays"]
        sorted_assays = sorted(
            assays, key=lambda x: len(x["description"]), reverse=False
        )

        print("number of assays before removing big ones", len(sorted_assays))
        sorted_assays = remove_big_assays(sorted_assays)
        print("number of assays before removing big ones", len(sorted_assays))

        need_new_assay = True
        assay_num = 0

        while sorted_assays:
            doc_len_dic = {"desc": 0, "name": 0, "var": 0}
            doc_len = 0
            curr_doc_ass_names = []
            curr_doc_ass_descs = []
            curr_doc_ass_vars = []
            tok_ass_vars = []

            while doc_len < context_length:
                print("descsuma", iter_get_num_tokens(curr_doc_ass_descs))
                print(doc_len_dic)
                if need_new_assay:
                    tok_ass_vars = []
                    try:
                        assay = sorted_assays.pop()
                        variables = assay["variables"]
                        assay_num += 1
                    except Exception as e:
                        print(e)
                        break
                    tok_ass_name = tokenizer(assay["name"])
                    tok_ass_desc = tokenizer(assay["description"])
                    tok_name_len = get_num_tokens(tok_ass_name)
                    tok_desc_len = get_num_tokens(tok_ass_desc)
                if doc_len == 0 or need_new_assay:
                    curr_doc_ass_names.append(tok_ass_name)
                    curr_doc_ass_descs.append(tok_ass_desc)
                    doc_len += tok_name_len + tok_desc_len
                    doc_len_dic["name"] += get_num_tokens(tok_ass_name)
                    doc_len_dic["desc"] += get_num_tokens(tok_ass_desc)
                    # print("doc_len after adding name + desc", doc_len_dic)
                    need_new_assay = False
                    continue

                if not variables:
                    # print("tok vars",type(tok_ass_vars))
                    curr_doc_ass_vars.append(tok_ass_vars)
                    tok_ass_vars = []
                    need_new_assay = True
                    # continue
                    pass
                else:
                    need_new_assay = False
                    var = variables.pop()
                    new_var = tokenizer(add_var_str(var))
                    doc_len += get_num_tokens(new_var)
                    doc_len_dic["var"] += get_num_tokens(new_var)
                    # print("doc_len after adding var", doc_len_dic)
                    # print("appended var",type(new_var))
                    tok_ass_vars.append(new_var)
            # print("curr doc vars",curr_doc_ass_vars)
            print("desc sumb", iter_get_num_tokens(curr_doc_ass_descs))
            print(doc_len_dic)
            if tok_ass_vars:
                # print("it was not empty")
                curr_doc_ass_vars.append(tok_ass_vars)
            # print("document length:", doc_len)
            difference = doc_len - context_length
            if difference > 0:
                sum = 0
                for a in curr_doc_ass_descs:
                    sum += get_num_tokens(a)
                name_sum = 0
                for b in curr_doc_ass_names:
                    name_sum += get_num_tokens(b)
                var_sum = 0
                for li in curr_doc_ass_vars:
                    for m in li:
                        var_sum += get_num_tokens(m)
                lein = len(curr_doc_ass_descs)
                print(
                    "description sum for document =",
                    iter_get_num_tokens(curr_doc_ass_descs),
                )

                curr_doc_ass_descs = evenly_remove_elements_from_lists(
                    curr_doc_ass_descs, difference
                )
                sum = 0
                for a in curr_doc_ass_descs:
                    sum += get_num_tokens(a)

                print("after desc", sum)
            curr_doc = []
            doc_len = 0
            assert len(curr_doc_ass_names) == len(curr_doc_ass_descs)
            # print("names", type(curr_doc_ass_names[0]))
            # print("descs", type(curr_doc_ass_descs[0]))
            # print("CRITICAL")
            curr_doc_ass_vars = list(itertools.chain.from_iterable(curr_doc_ass_vars))
            # print("vars", type(curr_doc_ass_vars[0]))
            batch_encoding = group_the_lists(
                [curr_doc_ass_names, curr_doc_ass_descs, curr_doc_ass_vars]
            )
            print("batch encoding length", len(batch_encoding["input_ids"]))
            # for i in range(len(curr_doc_ass_names)):
            #     curr_doc.extend(curr_doc_ass_names[i])
            #     curr_doc.extend(curr_doc_ass_descs[i])
            #     curr_doc.extend(curr_doc_ass_vars[i])

            if get_num_tokens(batch_encoding) == context_length:
                documents.append(batch_encoding)
            else:
                wrong_count += 1

            curr_doc_ass_names = []
            curr_doc_ass_descs = []
            curr_doc_ass_vars = []
            tok_ass_vars = []

            print("did a doc")
        break
print(wrong_count)
# for index,doc in enumerate(documents):
#     print(f"index {index} length {len(doc)}")
#     print(type(doc))
# for doc in documents:
#     print("------------------------------------------")
#     decoded_string = tokenizer.decode(doc['input_ids'])
#     print(decoded_string)
#     encoded = tokenizer.encode(decoded_string)

#     # Check the number of tokens
#     num_tokens = len(encoded)
#     print(num_tokens)
