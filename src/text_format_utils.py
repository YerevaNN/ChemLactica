import random
import time
from utils import get_tokenizer

SPECIAL_TAGS = {"SMILES": {"start": "[START_SMILES]", "end": "[END_SMILES]"}}


def delete_empty_tags(compound_json):
    for k, v in list(compound_json.items()):
        if v == []:
            del compound_json[k]
    return compound_json


def generate_formatted_string(compound_json):
    key_value_pairs = []
    if random.random() < 0.5:
        key = "SMILES"
        key_value_pairs.append(format_key_value(key, compound_json[key]))
        del compound_json["SMILES"]
    keys = list(compound_json.keys())
    for key in random.sample(keys, len(keys)):
        key_value_pairs.append(format_key_value(key, compound_json[key]))
    compound_formatted_string = "".join(key_value_pairs) + "</s>" # get_tokenizer().eos_token
    return compound_formatted_string
    # return str(compound_json)


def format_key_value(key, value):
    formatted_string = ""
    if key == "related":
        for pair in value:
            formatted_string += f"[SIMILAR {pair['similarity']} {pair['SMILES']}]"
    elif key == "experimental":
        for pair in value:
            formatted_string += f"[{pair['PROPERTY_NAME']} {pair['PROPERTY_VALUE']}]"
    elif key == "synonyms":
        for val in value:
            formatted_string += f"[SYNONYM {val['name']}]"
    else:
        if key in SPECIAL_TAGS:
            start = SPECIAL_TAGS[key]["start"]
            end = SPECIAL_TAGS[key]["end"]
            return f"{start}{value}{end}"
        formatted_string = f"[{key.upper()} {value}]"
        return formatted_string

    return formatted_string


def main():
    import json

    string = """{\"synonyms\":[{\"name\":\"p-Phenylazo carbanilic acid, n-hexyl ester\"}],
                 \"related\":[{\"SMILES\":\"CCCCOC(=O)NC1=CC=C(C=C1)N\",
                 \"similarity\":0.74}],\"experimental\":[],\"CID\":523129,
                 \"SMILES\":\"CCCCCCOC(=O)NC1=CC=C(C=C1)N=NC2=CC=CC=C2\",
                 \"SAS\":2.09,\"WEIGHT\":325.18,\"TPSA\":63.05,\"CLOGP\":6.23,
                 \"QED\":0.46,\"NUMHDONORS\":1,\"NUMHACCEPTORS\":4,
                 \"NUMHETEROATOMS\":5,\"NUMROTATABLEBONDS\":8,\"NOCOUNT\":5,
                 \"NHOHCOUNT\":1,\"RINGCOUNT\":2,\"HEAVYATOMCOUNT\":24,\"FRACTIONCSP3\":0.32,
                 \"NUMAROMATICRINGS\":2,\"NUMSATURATEDRINGS\":0,\"NUMAROMATICHETEROCYCLES\":0,
                 \"NUMAROMATICCARBOCYCLES\":2,\"NUMSATURATEDHETEROCYCLES\":0,
                 \"NUMSATURATEDCARBOCYCLES\":0,\"NUMALIPHATICRINGS\":0,
                 \"NUMALIPHATICHETEROCYCLES\":0,
                 \"NUMALIPHATICCARBOCYCLES\":0,
                 \"IUPAC\":\"hexyl N-(4-phenyldiazenylphenyl)carbamate\"}"""
    start_time = time.time()
    for i in range(100000):
        example = json.loads((string))
        example = delete_empty_tags(example)
        generate_formatted_string((example))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)


if __name__ == "__main__":
    main()
