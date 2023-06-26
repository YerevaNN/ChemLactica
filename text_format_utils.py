import random
import time

SPECIAL_TAGS = {"SMILES": {"start": "[START_SMILES] ", "end": " [END_SMILES]"}}


def generate_formatted_string(compound_json):
    key_value_pairs = list()
    for key in random.sample(list(compound_json.keys()), len(compound_json.keys())):
        key_value_pairs.append(format_key_value(key, compound_json[key]))
    compound_formatted_string = "".join(key_value_pairs)
    return compound_formatted_string


def format_key_value(key, value):
    if key in SPECIAL_TAGS:
        start = SPECIAL_TAGS[key]["start"]
        end = SPECIAL_TAGS[key]["end"]
        return f"{start}{value}{end}"
    else:
        upper_key = key.upper()
        return f"[{upper_key} {value}]"


def main():
    import json

    string = """{\"synonyms\":[{\"name\":\"p-Phenylazo carbanilic acid, n-hexyl ester\"}],\"related\":[{\"SMILES\":\"CCCCOC(=O)NC1=CC=C(C=C1)N\",\"similarity\":0.74}],\"experimental\":[],\"CID\":523129,\"SMILES\":\"CCCCCCOC(=O)NC1=CC=C(C=C1)N=NC2=CC=CC=C2\",\"SAS\":2.09,\"WEIGHT\":325.18,\"TPSA\":63.05,\"CLOGP\":6.23,\"QED\":0.46,\"NUMHDONORS\":1,\"NUMHACCEPTORS\":4,\"NUMHETEROATOMS\":5,\"NUMROTATABLEBONDS\":8,\"NOCOUNT\":5,\"NHOHCOUNT\":1,\"RINGCOUNT\":2,\"HEAVYATOMCOUNT\":24,\"FRACTIONCSP3\":0.32,\"NUMAROMATICRINGS\":2,\"NUMSATURATEDRINGS\":0,\"NUMAROMATICHETEROCYCLES\":0,\"NUMAROMATICCARBOCYCLES\":2,\"NUMSATURATEDHETEROCYCLES\":0,\"NUMSATURATEDCARBOCYCLES\":0,\"NUMALIPHATICRINGS\":0,\"NUMALIPHATICHETEROCYCLES\":0,\"NUMALIPHATICCARBOCYCLES\":0,\"IUPAC\":\"hexyl N-(4-phenyldiazenylphenyl)carbamate\"}"""
    start_time = time.time()
    for i in range(100000):
        example = json.loads((string))
        generate_formatted_string((example))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)


if __name__ == "__main__":
    main()
