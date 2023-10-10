import json
import random
from transformers import AutoTokenizer

jsonl_file_path = "/mnt/sxtn/phil/3953_start.jsonl"
tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m")

CONTEXT_LENGTH = 3000


def add_name(assay_object, document):
    document += "[ASSNAME" + assay_object["name"] + "]"
    return document


def add_var_str(var_object, document):
    document += "[VARNAME" + str(var_object["name"]) + "]"
    document += "[VARDESC" + str(var_object["description"]) + "]"
    document += "[VARVALUE" + str(var_object["value"]) + "]"
    return document


def remove_big_assays(assays):
    for assay in assays:
        if len(assay["description"]) > 1400:
            assays.remove(assay)
    return assays


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
        while sorted_assays:
            curr_doc = ""
            curr_doc_ass_names = []
            curr_doc_ass_descs = []
            curr_doc_ass_vars = []
            while len(curr_doc) < context_length:
                if need_new_assay:
                    try:
                        assay = sorted_assays.pop()
                    except Exception as e:
                        print(e)
                        break
                    curr_doc = add_name(assay, curr_doc)
                    recent_desc = assay["description"]
                    variables = random.shuffle(assay["variables"])
                if not variables:
                    need_new_assay = True
                    continue
                else:
                    need_new_assay = False
                    var = variables.pop()
                    curr_doc = add_var_str(var, curr_doc)

            while len(curr_doc) < context_length:
                curr_doc += "COMP_PROP_FILLER"
            # curr_doc = curr_doc[:3000]
            documents.append(curr_doc)
        break

for doc in documents:
    print(len(doc))
