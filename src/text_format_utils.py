import random
import time
from utils import get_tokenizer

SPECIAL_TAGS = {
    "SMILES": {"start": "[START_SMILES]", "end": "[END_SMILES]"},
    "synonym": {"start": "[SYNONYM]", "end": "[/SYNONYM]"},
    "RELATED": {"start": "[RELATED]", "end": "[/RELATED]"},
    "similarity": {"start": "[SIMILAR]", "end": "[/SIMILAR]", "type": float},
    "PROPERTY": {"start": "[PROPERTY]", "end": "[/PROPERTY]"},
    "SAS": {"start": "[SAS]", "end": "[/SAS]", "type": float},
    "WEIGHT": {"start": "[WEIGHT]", "end": "[/WEIGHT]", "type": float},
    "TPSA": {"start": "[TPSA]", "end": "[/TPSA]", "type": float},
    "CLOGP": {"start": "[CLOGP]", "end": "[/CLOGP]", "type": float},
    "QED": {"start": "[QED]", "end": "[/QED]", "type": float},
    "NUMHDONORS": {"start": "[NUMHDONORS]", "end": "[/NUMHDONORS]"},
    "NUMHACCEPTORS": {"start": "[NUMHACCEPTORS]", "end": "[/NUMHACCEPTORS]"},
    "NUMHETEROATOMS": {"start": "[NUMHETEROATOMS]", "end": "[/NUMHETEROATOMS]"},
    "NUMROTATABLEBONDS": {"start": "[NUMROTATABLEBONDS]", "end": "[/NUMROTATABLEBONDS]"},
    "NOCOUNT": {"start": "[NOCOUNT]", "end": "[/NOCOUNT]"},
    "NHOHCOUNT": {"start": "[NHOHCOUNT]", "end": "[/NHOHCOUNT]"},
    "RINGCOUNT": {"start": "[RINGCOUNT]", "end": "[/RINGCOUNT]"},
    "HEAVYATOMCOUNT": {"start": "[HEAVYATOMCOUNT]", "end": "[/HEAVYATOMCOUNT]"},
    "FRACTIONCSP3": {"start": "[FRACTIONCSP3]", "end": "[/FRACTIONCSP3]", "type": float},
    "NUMAROMATICRINGS": {"start": "[NUMAROMATICRINGS]", "end": "[/NUMAROMATICRINGS]"},
    "NUMSATURATEDRINGS": {"start": "[NUMSATURATEDRINGS]", "end": "[/NUMSATURATEDRINGS]"},
    "NUMAROMATICHETEROCYCLES": {"start": "[NUMAROMATICHETEROCYCLES]", "end": "[/NUMAROMATICHETEROCYCLES]"},
    "NUMAROMATICCARBOCYCLES": {"start": "[NUMAROMATICCARBOCYCLES]", "end": "[/NUMAROMATICCARBOCYCLES]"},
    "NUMSATURATEDHETEROCYCLES": {"start": "[NUMSATURATEDHETEROCYCLES]", "end": "[/NUMSATURATEDHETEROCYCLES]"},
    "NUMSATURATEDCARBOCYCLES": {"start": "[NUMSATURATEDCARBOCYCLES]", "end": "[/NUMSATURATEDCARBOCYCLES]"},
    "NUMALIPHATICRINGS": {"start": "[NUMALIPHATICRINGS]", "end": "[/NUMALIPHATICRINGS]"},
    "NUMALIPHATICHETEROCYCLES": {"start": "[NUMALIPHATICHETEROCYCLES]", "end": "[/NUMALIPHATICHETEROCYCLES]"},
    "NUMALIPHATICCARBOCYCLES": {"start": "[NUMALIPHATICCARBOCYCLES]", "end": "[/NUMALIPHATICCARBOCYCLES]"},
    "IUPAC": {"start": "[IUPAC]", "end": "[/IUPAC]"},
    "VAR_NAME": {"start": "[VAR_NAME]", "end": "[/VAR_NAME]"},
    "VAR_DESC": {"start": "[VAR_DESC]", "end": "[/VAR_DESC]"},
    "VAR_VAL": {"start": "[VAR_VAL]", "end": "[/VAR_VAL]"},
    "ASSAY_NAME": {"start": "[ASSAY_NAME]", "end": "[/ASSAY_NAME]"},
    "ASSAY_DESC": {"start": "[ASSAY_DESC]", "end": "[/ASSAY_DESC]"}
}


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


def format_key_value(key, value):
    if key == "CID": return ""
    formatted_string = ""
    if key == "related":
        for pair in value:
            formatted_string += f"{SPECIAL_TAGS['similarity']['start']}{pair['SMILES']} {pair['similarity']}{SPECIAL_TAGS['similarity']['end']}"
    elif key == "experimental":
        for pair in value:
            formatted_string += f"[PROPERTY]{pair['PROPERTY_NAME']} {pair['PROPERTY_VALUE']}[/PROPERTY]"
    elif key == "synonyms":
        for val in value:
            formatted_string += f"{SPECIAL_TAGS['synonym']['start']}{val['name']}{SPECIAL_TAGS['synonym']['end']}"
    else:
        try:
            if SPECIAL_TAGS[key].get('type') is float:
                value = "{:.2f}".format(float(value))
                assert len(value.split(".")[-1]) == 2
        except Exception as e:
            print(e)
        start = SPECIAL_TAGS[key]["start"]
        end = SPECIAL_TAGS[key]["end"]
        return f"{start}{value}{end}"

    return formatted_string


def main():
    import json

    string = """{\"synonyms\":[{\"name\":\"p-Phenylazo carbanilic acid, n-hexyl ester\"}],
                 \"related\":[{\"SMILES\":\"CCCCOC(=O)NC1=CC=C(C=C1)N\",
                 \"similarity\":0.7}],\"experimental\":[{\"PROPERTY_NAME\":\"Kovats Retention Index\",\"PROPERTY_VALUE\":\"Semi-standard non-polar: 4303\"},{\"PROPERTY_NAME\":\"Kovats Retention Index\",\"PROPERTY_VALUE\":\"Semi-standard non-polar: 4303\"}],\"CID\":523129,
                 \"SMILES\":\"CCCCCCOC(=O)NC1=CC=C(C=C1)N=NC2=CC=CC=C2\",
                 \"SAS\":2.0,\"WEIGHT\":325.1,\"TPSA\":63.06823,\"CLOGP\":6.231230123,
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
        print(generate_formatted_string((example)))
        break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)


if __name__ == "__main__":
    main()
