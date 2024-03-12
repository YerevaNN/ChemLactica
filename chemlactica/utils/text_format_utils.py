# import random
import time

# import os
# from functools import cache


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
    "NUMROTATABLEBONDS": {
        "start": "[NUMROTATABLEBONDS]",
        "end": "[/NUMROTATABLEBONDS]",
    },
    "NOCOUNT": {"start": "[NOCOUNT]", "end": "[/NOCOUNT]"},
    "NHOHCOUNT": {"start": "[NHOHCOUNT]", "end": "[/NHOHCOUNT]"},
    "RINGCOUNT": {"start": "[RINGCOUNT]", "end": "[/RINGCOUNT]"},
    "HEAVYATOMCOUNT": {"start": "[HEAVYATOMCOUNT]", "end": "[/HEAVYATOMCOUNT]"},
    "FRACTIONCSP3": {
        "start": "[FRACTIONCSP3]",
        "end": "[/FRACTIONCSP3]",
        "type": float,
    },
    "NUMAROMATICRINGS": {
        "start": "[NUMAROMATICRINGS]",
        "end": "[/NUMAROMATICRINGS]",
    },
    "NUMSATURATEDRINGS": {
        "start": "[NUMSATURATEDRINGS]",
        "end": "[/NUMSATURATEDRINGS]",
    },
    "NUMAROMATICHETEROCYCLES": {
        "start": "[NUMAROMATICHETEROCYCLES]",
        "end": "[/NUMAROMATICHETEROCYCLES]",
    },
    "NUMAROMATICCARBOCYCLES": {
        "start": "[NUMAROMATICCARBOCYCLES]",
        "end": "[/NUMAROMATICCARBOCYCLES]",
    },
    "NUMSATURATEDHETEROCYCLES": {
        "start": "[NUMSATURATEDHETEROCYCLES]",
        "end": "[/NUMSATURATEDHETEROCYCLES]",
    },
    "NUMSATURATEDCARBOCYCLES": {
        "start": "[NUMSATURATEDCARBOCYCLES]",
        "end": "[/NUMSATURATEDCARBOCYCLES]",
    },
    "NUMALIPHATICRINGS": {
        "start": "[NUMALIPHATICRINGS]",
        "end": "[/NUMALIPHATICRINGS]",
    },
    "NUMALIPHATICHETEROCYCLES": {
        "start": "[NUMALIPHATICHETEROCYCLES]",
        "end": "[/NUMALIPHATICHETEROCYCLES]",
    },
    "NUMALIPHATICCARBOCYCLES": {
        "start": "[NUMALIPHATICCARBOCYCLES]",
        "end": "[/NUMALIPHATICCARBOCYCLES]",
    },
    "IUPAC": {"start": "[IUPAC]", "end": "[/IUPAC]"},
    "VAR_NAME": {"start": "[VAR_NAME]", "end": "[/VAR_NAME]"},
    "VAR_DESC": {"start": "[VAR_DESC]", "end": "[/VAR_DESC]"},
    "VAR_VAL": {"start": "[VAR_VAL]", "end": "[/VAR_VAL]"},
    "ASSAY_NAME": {"start": "[ASSAY_NAME]", "end": "[/ASSAY_NAME]"},
    "ASSAY_DESC": {"start": "[ASSAY_DESC]", "end": "[/ASSAY_DESC]"},
}


def delete_empty_tags(compound_json):
    for k, v in list(compound_json.items()):
        if v == [] or v == "":
            del compound_json[k]
    return compound_json


def generate_formatted_string(compound_json, rng):
    key_value_pairs = []
    if compound_json.get("SMILES") and rng.random() < 0.5:
        key = "SMILES"
        key_value_pairs.append(format_key_value(key, compound_json[key], rng))
        del compound_json["SMILES"]
    keys = list(compound_json.keys())
    rng.shuffle(keys)

    for key in keys:
        key_value_pairs.append(format_key_value(key, compound_json[key], rng))
    compound_formatted_string = (
        "".join(key_value_pairs) + "</s>"
    )  # get_tokenizer().eos_token
    return compound_formatted_string


def format_key_value(key, value, rng):
    if key == "CID":
        return ""
    formatted_string = ""
    if key == "related":
        if len(value) > 10:
            # value = random.sample(value, 5)
            value = rng.choice(value, size=5, replace=False, shuffle=False)
        for pair in value:
            formatted_string += f"{SPECIAL_TAGS['similarity']['start']}{pair['SMILES']} {pair['similarity']}{SPECIAL_TAGS['similarity']['end']}"  # noqa
    elif key == "experimental":
        for pair in value:
            formatted_string += (
                f"[PROPERTY]{pair['PROPERTY_NAME']} {pair['PROPERTY_VALUE']}[/PROPERTY]"
            )
    elif key == "synonyms":
        for val in value:
            formatted_string += f"{SPECIAL_TAGS['synonym']['start']}{val['name']}{SPECIAL_TAGS['synonym']['end']}"  # noqa
    else:
        try:
            if SPECIAL_TAGS[key].get("type") is float:
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

    # string = """{\"synonyms\":[{\"name\":\"p-Phenylazo carbanilic acid, n-hexyl ester\"}],
    #              \"related\":[{\"SMILES\":\"CCCCOC(=O)NC1=CC=C(C=C1)N\",
    #              \"similarity\":0.7}],\"experimental\":[{\"PROPERTY_NAME\":
    #              \"Kovats Retention Index\",
    #              \"PROPERTY_VALUE\":\"Semi-standard non-polar: 4303\"},{\"PROPERTY_NAME\":
    #              \"Kovats Retention Index\",\"PROPERTY_VALUE\":
    #              \"Semi-standard non-polar: 4303\"}],
    #              \"CID\":523129,
    #              \"SMILES\":\"CCCCCCOC(=O)NC1=CC=C(C=C1)N=NC2=CC=CC=C2\",
    #              \"SAS\":2.0,\"WEIGHT\":325.1,\"TPSA\":63.06823,\"CLOGP\":6.231230123,
    #              \"QED\":0.46,\"NUMHDONORS\":1,\"NUMHACCEPTORS\":4,
    #              \"NUMHETEROATOMS\":"",\"NUMROTATABLEBONDS\":8,\"NOCOUNT\":5,
    #              \"NHOHCOUNT\":1,\"RINGCOUNT\":2,\"HEAVYATOMCOUNT\":24,\"FRACTIONCSP3\":0.32,
    #              \"NUMAROMATICRINGS\":2,\"NUMSATURATEDRINGS\":0,\"NUMAROMATICHETEROCYCLES\":0,
    #              \"NUMAROMATICCARBOCYCLES\":2,\"NUMSATURATEDHETEROCYCLES\":0,
    #              \"NUMSATURATEDCARBOCYCLES\":0,\"NUMALIPHATICRINGS\":0,
    #              \"NUMALIPHATICHETEROCYCLES\":0,
    #              \"NUMALIPHATICCARBOCYCLES\":0,
    #              \"IUPAC\":\"hexyl N-(4-phenyldiazenylphenyl)carbamate\"}"""
    string = """{\"synonyms\":[{\"name\":\"2,4-D-butotyl\"},{\"name\":\"1929-73-3\"},{\"name\":\"Aqua-Kleen\"},{\"name\":\"2,4-D butoxyethyl ester\"},{\"name\":\"Brush killer 64\"},{\"name\":\"2-Butoxyethyl 2,4-dichlorophenoxyacetate\"},{\"name\":\"Weedone LV 4\"},{\"name\":\"Bladex-B\"},{\"name\":\"2,4-D butoxyethanol ester\"},{\"name\":\"2,4-D 2-Butoxyethyl ester\"},{\"name\":\"Butoxyethyl (2,4-dichlorophenoxy)acetate\"},{\"name\":\"2-butoxyethyl 2-(2,4-dichlorophenoxy)acetate\"},{\"name\":\"2,4-D-butotyl [ISO]\"},{\"name\":\"2,4-D-Bee\"},{\"name\":\"2,4-DBEE\"},{\"name\":\"Acetic acid, (2,4-dichlorophenoxy)-, 2-butoxyethyl ester\"},{\"name\":\"2,4-D-butylglycol ester\"},{\"name\":\"(2,4-Dichlorophenoxy)acetic acid butoxyethyl ester\"},{\"name\":\"2,4-Dichlorophenoxyacetic acid butoxyethanol ester\"},{\"name\":\"Y4OL636NHU\"},{\"name\":\"Acetic acid, (2,4-dichlorophenoxy)-, butoxyethyl ester\"},{\"name\":\"Lo-Estasol\"},{\"name\":\"2-Butoxyethyl-2-(2,4-dichlorophenoxy)acetate\"},{\"name\":\"2,4-Dichlorophenoxyacetic acid Butoxyethyl Ester\"},{\"name\":\"Weedone LV-6\"},{\"name\":\"(2,4-Dichlorophenoxy)acetic acid butoxyethanol ester\"},{\"name\":\"Butoxy-D 3\"},{\"name\":\"Weed-Rhap LV-4D\"},{\"name\":\"Weedone 638\"},{\"name\":\"2,4-D butoxyethanol\"},{\"name\":\"Caswell No. 315AI\"},{\"name\":\"2,4-D-(2-Butoxyethyl)\"},{\"name\":\"HSDB 6307\"},{\"name\":\"EINECS 217-680-1\"},{\"name\":\"Butoxyethyl 2,4-dichlorophenoxyacetate\"},{\"name\":\"UNII-Y4OL636NHU\"},{\"name\":\"EPA Pesticide Chemical Code 030053\"},{\"name\":\"BRN 1996617\"},{\"name\":\"Butoxyethanol ester of 2,4-dichlorophenoxyacetic acid\"},{\"name\":\"CCRIS 8562\"},{\"name\":\"2,4-Dichlorophenoxyacetic acid 2-Butoxyethyl ester\"},{\"name\":\"2,4-D Butoxyethyl\"},{\"name\":\"Butoxyethanol ester of (2,4-dichlorophenoxy)acetic acid\"},{\"name\":\"BLADEX B\"},{\"name\":\"2,4-Dichlorophenoxyacetic acid ethylene glycol butyl ether ester\"},{\"name\":\"DESORMONE LOURD D\"},{\"name\":\"DESORMONE 600\"},{\"name\":\"DSSTox_CID_12309\"},{\"name\":\"DSSTox_RID_78911\"},{\"name\":\"DSSTox_GSID_32309\"},{\"name\":\"4-06-00-00911 (Beilstein Handbook Reference)\"},{\"name\":\"Butoxyethanol ester of 2,4-D\"},{\"name\":\"SCHEMBL2464242\"},{\"name\":\"CHEMBL3186113\"},{\"name\":\"DTXSID1032309\"},{\"name\":\"2,4-D-BUTOTYL [HSDB]\"},{\"name\":\"2,4-D, BUTOXYETHYL ESTER\"},{\"name\":\"ZINC2039104\"},{\"name\":\"Tox21_301559\"},{\"name\":\"AKOS015913890\"},{\"name\":\"NCGC00255339-01\"},{\"name\":\"CAS-1929-73-3\"},{\"name\":\"2,4-dichloro-phenoxyacetic acid butoxyethyl ester\"},{\"name\":\"929D733\"},{\"name\":\"W-109728\"},{\"name\":\"(2,4-Dichlorophenoxy)acetic acid 2-butoxyethyl ester\"},{\"name\":\"Q22329266\"},{\"name\":\"2,4-D butylglycol ester, PESTANAL(R), analytical standard\"},{\"name\":\"ACETIC ACID, 2-(2,4-DICHLOROPHENOXY)-, 2-BUTOXYETHYL ESTER\"},{\"name\":\"2,4-D 2-Butoxyethyl Ester ((2,4-Dichlorophenoxy)acetic Acid 2-Butoxyethyl Ester)\"}],\"related\":[{\"SMILES\":\"COP(=O)(C(C(Cl)(Cl)Cl)OC(=O)COC1=C(C=C(C=C1)Cl)Cl)OC\",\"similarity\":0.57},{\"SMILES\":\"C1=CC(=C(C=C1Cl)Cl)OCC(=O)OCOC(=O)Cl\",\"similarity\":0.7},{\"SMILES\":\"CC(C)(C)OC(=O)COC1=CC=CC=C1Cl\",\"similarity\":0.62},{\"SMILES\":\"C1=CC(=C(C=C1Cl)Cl)OCCOC(=O)C(=O)OCCOC2=C(C=C(C=C2)Cl)Cl\",\"similarity\":0.74},{\"SMILES\":\"CCOC(=O)C=C(C)OC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.62},{\"SMILES\":\"COC(=O)/C=C/C(=O)OCC(=O)OC1=C(C=C(C=C1Cl)Cl)Cl\",\"similarity\":0.69},{\"SMILES\":\"CCOC(=O)C(OC(=O)COC1=C(C=C(C=C1)Cl)Cl)OC(=O)Cl\",\"similarity\":0.74},{\"SMILES\":\"CCC(=O)OC(C(CCl)OC1=C(C=C(C=C1)Cl)Cl)Cl\",\"similarity\":0.72},{\"SMILES\":\"CCCCOC(C)COC(=O)COC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.92},{\"SMILES\":\"CCCCOC(=O)COC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.94},{\"SMILES\":\"CCOC(=O)COCC(=O)OC1=CC=C(C=C1)Cl\",\"similarity\":0.76},{\"SMILES\":\"[2H]C1=C(C(=C(C(=C1OCC(=O)O)Cl)[2H])Cl)[2H]\",\"similarity\":0.66},{\"SMILES\":\"C=CC(=O)[O-].C1=CC(=C(C=C1Cl)Cl)OCC(=O)[O-].[Cu+2]\",\"similarity\":0.6},{\"SMILES\":\"CCOC(=O)COC1=CC=C(C=C1)Cl\",\"similarity\":0.73},{\"SMILES\":\"[Li+].C1=CC(=C(C=C1Cl)Cl)OCC(=O)[O-]\",\"similarity\":0.62},{\"SMILES\":\"C[C@H](C(=O)OCCOC)OC1=CC=C(C=C1)OC2=C(C=C(C=C2)Cl)Cl\",\"similarity\":0.8},{\"SMILES\":\"CC(=C)C(=O)OC(COC1=C(C=C(C=C1)Cl)Cl)COC(=O)C=C\",\"similarity\":0.67},{\"SMILES\":\"CC(C(=O)[O-])OC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.59},{\"SMILES\":\"C[C@@H](C(=O)OC)OC1=C(C(=C(C=C1)Cl)OC)Cl\",\"similarity\":0.57},{\"SMILES\":\"CCOC(=O)C(=C(C)OC)OC1=CC=C(C=C1)Cl\",\"similarity\":0.6},{\"SMILES\":\"CC(C)CCCOC(=O)COCC(=O)OC1=CC=CC=C1Cl\",\"similarity\":0.88},{\"SMILES\":\"CCCCC(CC)COC(=O)COC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.89},{\"SMILES\":\"CCCCOCCOCCCOC(=O)COC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.97},{\"SMILES\":\"CC(C)OC(=O)CCCOC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.69},{\"SMILES\":\"C[C@@H](C(=O)O)OC1=C(C=CC=C1Cl)Cl\",\"similarity\":0.59},{\"SMILES\":\"CC(C)CCCOC(=O)COC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.85},{\"SMILES\":\"CCCCOC(=O)COC(=O)COC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.95},{\"SMILES\":\"CC[Si](C)(C)COC(=O)COC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.71},{\"SMILES\":\"C1=CC(=C(C=C1Cl)Cl)OCC(=O)OCC#CI\",\"similarity\":0.71},{\"SMILES\":\"CCCOC(=O)COCC(=O)OC1=CC=C(C=C1)Cl\",\"similarity\":0.89},{\"SMILES\":\"C=CC(=O)[O-].C1=CC(=C(C=C1Cl)Cl)OCC(=O)[O-].[Mg+2]\",\"similarity\":0.6},{\"SMILES\":\"C1=CC(=C(C=C1Cl)Cl)OCC(=O)[O-].C1=CC(=C(C=C1Cl)Cl)OCC(=O)[O-].[Ca+2]\",\"similarity\":0.63},{\"SMILES\":\"C1=CC(=C(C=C1Cl)Cl)OCC(=O)OCC#CCOC(=O)COC2=C(C=C(C=C2)Cl)Cl\",\"similarity\":0.7},{\"SMILES\":\"CC(=O)OCCOC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.81},{\"SMILES\":\"CC(C(=O)OC1CCCCC1)(OC2=CC=CC=C2)OC3=C(C=C(C=C3)Cl)Cl\",\"similarity\":0.72},{\"SMILES\":\"CCC(C(=O)OCC)OC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.76},{\"SMILES\":\"CC(C)(COC(=O)C(=C)OC1=C(C=C(C=C1)Cl)Cl)COC(=O)C(=C)OC2=C(C=C(C=C2)Cl)Cl\",\"similarity\":0.6},{\"SMILES\":\"C1=C(C=C(C(=C1Cl)OC(C(=O)O)F)Cl)Cl\",\"similarity\":0.55},{\"SMILES\":\"CCOC(=O)C(OC1=CC=CC=C1Cl)(F)F\",\"similarity\":0.67},{\"SMILES\":\"CC(C)COC(C)COC(=O)COC1=C(C=C(C=C1)Cl)Cl\",\"similarity\":0.79}],\"experimental\":[{\"PROPERTY_NAME\":\"Metabolism/Metabolites\",\"PROPERTY_VALUE\":\"Plants hydrolyze 2,4-D esters to 2,4-D, which is the active herbicide. ... Further metabolism ... occurs through three mechanisms, namely, side chain degradation, hydroxylation of the aromatic ring, and conjugation with plant constituents. /2,4-D esters/\"},{\"PROPERTY_NAME\":\"Metabolism/Metabolites\",\"PROPERTY_VALUE\":\"HERBICIDAL ACTIVITY OF ESTERS, NITRILES, AMINES (&, OF COURSE, SALTS) APPEARS SIMILAR IF NOT IDENTICAL TO PARENT ACID. THIS IS APPARENTLY DUE TO PRESENCE OF HYDROLYTIC ENZYMES IN PLANTS & IN SOIL MICROORGANISMS THAT CONVERT THESE DERIVATIVES TO PARENT ACID. /2,4-D ESTER/\"},{\"PROPERTY_NAME\":\"Metabolism/Metabolites\",\"PROPERTY_VALUE\":\"2,4-D ESTERS ARE HYDROLYZED IN ANIMALS. THE PHENOXY ACIDS ARE EXCRETED PREDOMINANTLY AS SUCH IN THE URINE OF RATS AFTER THEIR ORAL ADMIN, ALTHOUGH MINOR PORTION IS CONJUGATED WITH AMINO ACIDS GLYCINE & TAURINE & WITH GLUCURONIC ACID. /2,4-D AND ESTERS/\"},{\"PROPERTY_NAME\":\"Metabolism/Metabolites\",\"PROPERTY_VALUE\":\"Metabolites of 2,4-D other than conjugates have not been detected in human urine.\"},{\"PROPERTY_NAME\":\"Solubility\",\"PROPERTY_VALUE\":\"ESTER FORMULATIONS HAVE LOW SOLUBILITY IN WATER. /2,4-D; SRP: THEY CAN BE DISPERSED AS AQUEOUS EMULSIONS/\"},{\"PROPERTY_NAME\":\"Solubility\",\"PROPERTY_VALUE\":\"Sol in organic solvents @ 20 \u00b0C; miscible in acetone, acetonitrile, n-hexane, and methanol.\"},{\"PROPERTY_NAME\":\"Solubility\",\"PROPERTY_VALUE\":\"In water: 12 mg/l, @ 25 \u00b0C\"},{\"PROPERTY_NAME\":\"Solubility\",\"PROPERTY_VALUE\":\"Soluble in oils /2,4-D esters/\"},{\"PROPERTY_NAME\":\"Physical Description\",\"PROPERTY_VALUE\":\"Colorless to amber liquid; Insoluble in water (12 mg/L at 25 deg C); [HSDB] Clear liquid; Insoluble in water; [MSDSonline] Dark amber liquid; [Reference #1\"},{\"PROPERTY_NAME\":\"Boiling Point\",\"PROPERTY_VALUE\":\"156-162 \u00b0C @ 1-1.5 mm Hg\"},{\"PROPERTY_NAME\":\"Melting Point\",\"PROPERTY_VALUE\":\"Liquid at room temperature\"},{\"PROPERTY_NAME\":\"Chemical Classes\",\"PROPERTY_VALUE\":\"Pesticides -> Herbicides, Chlorophenoxy\"},{\"PROPERTY_NAME\":\"Vapor Pressure\",\"PROPERTY_VALUE\":\"4.5X10-6 mm Hg @ 25 \u00b0C\"},{\"PROPERTY_NAME\":\"Vapor Pressure\",\"PROPERTY_VALUE\":\"0.0000045 [mmHg\"},{\"PROPERTY_NAME\":\"Decomposition\",\"PROPERTY_VALUE\":\"When heated to decomp, it emits toxic fumes of /hydrogen chloride /.\"},{\"PROPERTY_NAME\":\"Decomposition\",\"PROPERTY_VALUE\":\"When heated to decomposition, emits highly toxic vapors. /2,4-D esters/\"},{\"PROPERTY_NAME\":\"Odor\",\"PROPERTY_VALUE\":\"Odorless (when pure)\"},{\"PROPERTY_NAME\":\"Other Experimental Properties\",\"PROPERTY_VALUE\":\"Decomp temp greater than 200 \u00b0C at 760 mm Hg\"},{\"PROPERTY_NAME\":\"Other Experimental Properties\",\"PROPERTY_VALUE\":\"VAP: 40.5% and 60% ester had vap press of 3.64 and 2.67x10-6 mm Hg at 25 \u00b0C\"},{\"PROPERTY_NAME\":\"Other Experimental Properties\",\"PROPERTY_VALUE\":\"/2,4-D esters/ are generally immiscible or insoluble in water, but gradual hydrolysis /SRP: will occur in very alkaline waters/. /2,4-D esters/\"},{\"PROPERTY_NAME\":\"Other Experimental Properties\",\"PROPERTY_VALUE\":\"Fuel oil-like odor /2,4-D esters/ SRP: Technical product.\"},{\"PROPERTY_NAME\":\"Other Experimental Properties\",\"PROPERTY_VALUE\":\"SRP: 2,4-D ESTERS ARE SOLUBLE IN NON-POLAR ORGANIC SOLVENTS SUCH AS HEXANE, BENZENE, ACETONE, AND ALCOHOLS. /2,4-D ESTERS/\"},{\"PROPERTY_NAME\":\"Flash Point\",\"PROPERTY_VALUE\":\"Greater than 175 \u00b0F (open cup) /2,4-D esters/\"},{\"PROPERTY_NAME\":\"Stability/Shelf Life\",\"PROPERTY_VALUE\":\"Shelf life of ester formulations varies, depending on the emulsifying system. Some retain satisfactory emulsifying properties after 3 yr. /2,4-D ester formulations/\"},{\"PROPERTY_NAME\":\"Color/Form\",\"PROPERTY_VALUE\":\"Amber liquid\"},{\"PROPERTY_NAME\":\"Color/Form\",\"PROPERTY_VALUE\":\"Viscous, colorless liquid\"},{\"PROPERTY_NAME\":\"Density\",\"PROPERTY_VALUE\":\"1.232 g/cu cm @ 20 \u00b0C\"},{\"PROPERTY_NAME\":\"GC-MS\",\"PROPERTY_VALUE\":\"m/z Top Peak: 57\"},{\"PROPERTY_NAME\":\"GC-MS\",\"PROPERTY_VALUE\":\"m/z 2nd Highest: 29\"},{\"PROPERTY_NAME\":\"GC-MS\",\"PROPERTY_VALUE\":\"m/z 3rd Highest: 41\"},{\"PROPERTY_NAME\":\"GC-MS\",\"PROPERTY_VALUE\":\"m/z Top Peak: 57\"},{\"PROPERTY_NAME\":\"GC-MS\",\"PROPERTY_VALUE\":\"m/z 2nd Highest: 56\"},{\"PROPERTY_NAME\":\"GC-MS\",\"PROPERTY_VALUE\":\"m/z 3rd Highest: 41\"},{\"PROPERTY_NAME\":\"Non-Human Toxicity Excerpts\",\"PROPERTY_VALUE\":\"Chicks (n= 10) of mixed sexes fed 0-1000 mg/kg for 21 days, no effect on growth rate /was observed/. Chicks (n= 10) dosed at 2,000-7,500 mg/kg for 21 days, /toxic symptoms included:/ reduced food intake and growth; swollen kidneys. Chicks (n= 10) fed 5000 mg/kg for 7 days, exhibited reduced growth rate. Seven to 14 days after /cessation of the ingestion of the compound/ normal growth resumed.\"},{\"PROPERTY_NAME\":\"Non-Human Toxicity Excerpts\",\"PROPERTY_VALUE\":\"Mutagenic effects: Studies available at present are not adequate for the quantitive evaluation of the mutagenic effects of 2,4-D and its derivatives in short-term tests. However, the evidence does not suggest that 2,4-D derivatives are potent mutagens. /2,4-D and its derivatives/\"},{\"PROPERTY_NAME\":\"Non-Human Toxicity Excerpts\",\"PROPERTY_VALUE\":\"Chlorophenoxy herbicides are classified as vasculotoxic agents with the associated disease state of hypertension. /Chlorophenoxy herbicides, from table/\"},{\"PROPERTY_NAME\":\"Non-Human Toxicity Excerpts\",\"PROPERTY_VALUE\":\"Pathologic changes in exptl animals killed by the chlorophenoxy compounds are generally nonspecific with irritation of stomach & some liver & kidney injury. /Chlorophenoxy compounds/\"},{\"PROPERTY_NAME\":\"Non-Human Toxicity Excerpts\",\"PROPERTY_VALUE\":\"For more Non-Human Toxicity Excerpts (Complete) data for 2,4-D BUTOXYETHYL ESTER (8 total), please visit the HSDB record page.\"},{\"PROPERTY_NAME\":\"Non-Human Toxicity Values\",\"PROPERTY_VALUE\":\"LD50 Rat male oral 940 mg/kg\"},{\"PROPERTY_NAME\":\"Non-Human Toxicity Values\",\"PROPERTY_VALUE\":\"LD50 Rabbit percutaneous in the range of 4000 mg/kg\"},{\"PROPERTY_NAME\":\"Non-Human Toxicity Values\",\"PROPERTY_VALUE\":\"LD50 Chicks 4-week old oral. Acid equivalent was 588 mg/kg.\"},{\"PROPERTY_NAME\":\"Non-Human Toxicity Values\",\"PROPERTY_VALUE\":\"LD50 Rat oral 831 mg/kg\"},{\"PROPERTY_NAME\":\"Volatilization from Water/Soil\",\"PROPERTY_VALUE\":\"The Henry's Law constant for 2,4-D, butoxyethyl ester is estimated as 1.6X10-7 atm-cu m/mole(SRC) derived from its vapor pressure, 4.5X10-6 mm Hg(1), and water solubility, 12 mg/l(2). This Henry's Law constant indicates that 2,4-D butoxyethyl ester is expected to be essentially nonvolatile from water surfaces(3). 2,4-D butoxyethyl ester is not expected to volatilize from dry soil surfaces(SRC) based upon its vapor pressure(1).\"},{\"PROPERTY_NAME\":\"Soil Adsorption/Mobility\",\"PROPERTY_VALUE\":\"The Koc of 2,4-D butoxyethyl ester is estimated as 1,100(SRC), using a water solubility of 12 mg/l(1) and a regression-derived equation(2). According to a classification scheme(3), this estimated Koc value suggests that 2,4-D butoxyethyl ether is expected to have low mobility in soil.\"},{\"PROPERTY_NAME\":\"Probable Routes of Human Exposure\",\"PROPERTY_VALUE\":\"2,4-D and its derivatives can be absorbed via the oral, dermal, and inhalation routes. General population exposure is mainly by the oral route, but under occupational and bystander exposure conditions, the dermal route is by far the most important. /2,4-D and its derivatives/\"},{\"PROPERTY_NAME\":\"Probable Routes of Human Exposure\",\"PROPERTY_VALUE\":\"Occupational exposure to 2,4-D butoxyethyl ester may occur through inhalation and dermal contact with this compound at workplaces where 2,4-D butoxyethyl ester is produced or used. Agricultural workers may be exposed to 2,4-D butoxyethyl ester during spraying operation using herbicides containing the chemical. Exposure will be mainly by inhalation and dermal contact. Monitoring data indicate that the general population may be exposed to 2,4-D butoxyethyl ester via inhalation of spray drift in areas adjacent to spraying. (SRC)\"},{\"PROPERTY_NAME\":\"Environmental Fate/Exposure Summary\",\"PROPERTY_VALUE\":\"2,4-D butoxyethyl's production may result in its release to the environment through various waste streams; it's use as a herbicide will result in its direct release to the environment. If released to air, a vapor pressure of 4.5X10-6 mm Hg at 25 \u00b0C indicates 2,4-D butoxyethyl ester will exist in both the vapor and particulate phases in the ambient atmosphere. Vapor-phase 2,4-D butoxyethyl ester will be degraded in the atmosphere by reaction with photochemically-produced hydroxyl radicals; the half-life for this reaction in air is estimated to be 15 hrs. Particulate-phase 2,4-D butoxyethyl ester will be removed from the atmosphere by wet and dry deposition. 2,4-D butoxyethyl ester absorbs UV radiation >290 nm up to approximately 320 nm and undergoes direct photolysis. If released to soil, 2,4-D butoxyethyl ester is expected to have low mobility based upon an estimated Koc of 1,100. Volatilization from moist soil surfaces is not expected to be an important fate process based upon an estimated Henry's Law constant of 1.6X10-7 atm-cu m/mole. If released into water, 2,4-D butoxyethyl ester is expected to adsorb to suspended solids and sediment based upon the estimated Koc. Half-lives obtained from batch cultures of grab samples using periphyton ecosystems ranged from 0.4 to 3 hr, indicating that biodegradation in water is an important environmental fate process. Volatilization from water surfaces is not expected to be an important fate process based upon this compound's estimated Henry's Law constant. BCF values of 7-55 suggest bioconcentration in aquatic organisms is low to moderate. The hydrolysis of the 2,4-D butoxyethyl ester is catalyzed by base and acid with the minimum rate of hydrolysis occuring just below pH 4. The calculated half-life at 28 \u00b0C decreases from 26 days at pH 6 to 0.6 hr at pH 9. Occupational exposure to 2,4-D butoxyethyl ester may occur through inhalation and dermal contact with this compound at workplaces where 2,4-D butoxyethyl ester is produced or used. Monitoring data indicate that the general population may be exposed to 2,4-D butoxyethyl ester via inhalation of spray drift in areas adjacent to spraying. (SRC)\"},{\"PROPERTY_NAME\":\"Environmental Fate\",\"PROPERTY_VALUE\":\"Various amounts of 2,4-D products applied to a target area may be distributed in the general environment, within a few hours or days, by the movements of air, water, or soil, particularly during periods of rain, high winds, or high temperature. Persistence or accumulation of 2,4-D residues from normal use is occasionally possible, mainly under dry or cold conditions where there is little biological activity. /2,4-D products/\"},{\"PROPERTY_NAME\":\"Environmental Fate\",\"PROPERTY_VALUE\":\"AQUATIC FATE: Persistence in aquatic systems depends on the water type, organic particulate matter, rain, sunlight, temperature, microbial degradation, volatilization, and oxygen content of the water. Accumulation in bottom sediments may also be a factor, but in general, not for the phenoxys. Microbial activity is the major means for detoxification of the phenoxys in soils, but is relatively unimportant in natural waters, but dominates in bottom mud sediments and in sludge.\"},{\"PROPERTY_NAME\":\"Environmental Fate\",\"PROPERTY_VALUE\":\"TERRESTRIAL FATE: From limited data available, it may be concluded that any phenoxy herbicide, whether applied as ester ... formulations, may be chemically transformed to the same phenoxyalkanoic anion in soil and water at rates dependent on pH. These anions would presumably reassociate with a variety of inorganic cations present in the soil to maintain electrical neutrality, and then undergo leaching and biological degradation. /Phenoxy esters/\"},{\"PROPERTY_NAME\":\"Environmental Fate\",\"PROPERTY_VALUE\":\"TERRESTRIAL FATE: Based on a classification scheme(1), an estimated Koc value of 1,100(SRC), determined from a water solubility of 12 mg/l(2) and a regression-derived equation(3), indicates that 2,4-D butoxyethyl ester is expected to have low mobility in soil(SRC). Volatilization of 2,4-D butoxyethyl ether from moist soil surfaces is not expected to be an important fate process(SRC) given an estimated Henry's Law constant of 1.6X10-7 atm-cu m/mole(SRC), derived from its vapor pressure, 4.5X10-6 mm Hg(4), and water solubility(2). 2,4-D butoxyethyl ester is not expected to volatilize from dry soil surfaces(SRC) based upon a its vapor pressure(4). Half-lives obtained from batch cultures of grab samples using periphyton ecosystems taken from 4 field sites ranged from 0.4 to 3 hr(5), indicating that biodegradation in soil may be an important environmental fate process(SRC).\"},{\"PROPERTY_NAME\":\"Environmental Fate\",\"PROPERTY_VALUE\":\"For more Environmental Fate (Complete) data for 2,4-D BUTOXYETHYL ESTER (7 total), please visit the HSDB record page.\"},{\"PROPERTY_NAME\":\"Environmental Bioconcentration\",\"PROPERTY_VALUE\":\"Channel catfish, and bluegills exposed to radiolabeled 2,4-D butoxyethyl ester rapidly took up the chemical showing maximum concentrations within 1 to 2 hr exposure for fed fish and 2 to 6 hr for fasted fish(1). Measured BCFs of 7-55 and 20-55 were obtained for fed and fasted fish, respectively(1). According to a classification scheme(2), these BCF values suggest the potential for bioconcentration in aquatic organisms is low to moderate(SRC). The fish convert the ester to the free acid which is rapidly excreted; residues in most tissues and organs declined and exponentially approached negligible concentrations(1).\"},{\"PROPERTY_NAME\":\"Environmental Abiotic Degradation\",\"PROPERTY_VALUE\":\"Calculated sunlight photolysis half-lives of butoxyethyl ester, at latitude 34 deg N ranged from 59 hr in summer to 430 hr in winter ...\"},{\"PROPERTY_NAME\":\"Environmental Abiotic Degradation\",\"PROPERTY_VALUE\":\"The hydrolysis half-life of the butoxyethyl ester at 25 \u00b0C incr from 9 hr at pH 8 to more than 1 yr at pH 5.\"},{\"PROPERTY_NAME\":\"Environmental Abiotic Degradation\",\"PROPERTY_VALUE\":\"/2,4-D esters, irradiated with wavelengths longer than 290 nm/ in organic solvents or at high concentrations in water, ... yielded ... 2-, and 4-chlorophenoxyacetic acid esters. Ester photolysis rates were also found to be pH independent. /2,4-D esters/\"},{\"PROPERTY_NAME\":\"Environmental Abiotic Degradation\",\"PROPERTY_VALUE\":\"Sunlight falling on the earth's surface is composed of wavelengths greater than about 280 nm ... phenoxy herbicides have an ultraviolet absorption maxima in water in the 280-290 nm range, they ... can absorb radiation and be photochemically degraded. /Phenoxy herbicides/\"},{\"PROPERTY_NAME\":\"Environmental Abiotic Degradation\",\"PROPERTY_VALUE\":\"For more Environmental Abiotic Degradation (Complete) data for 2,4-D BUTOXYETHYL ESTER (10 total), please visit the HSDB record page.\"},{\"PROPERTY_NAME\":\"Artificial Pollution Sources\",\"PROPERTY_VALUE\":\"2,4-D butoxyethyl's production may result in its release to the environment through various waste streams; it's use as a herbicide(1) will result in its direct release to the environment(SRC).\"}],\"CID\":16002,\"SMILES\":\"CCCCOCCOC(=O)COC1=C(C=C(C=C1)Cl)Cl\",\"SAS\":1.92,\"WEIGHT\":320.06,\"TPSA\":44.76,\"CLOGP\":3.73,\"QED\":0.51,\"NUMHDONORS\":0,\"NUMHACCEPTORS\":4,\"NUMHETEROATOMS\":6,\"NUMROTATABLEBONDS\":9,\"NOCOUNT\":4,\"NHOHCOUNT\":0,\"RINGCOUNT\":1,\"HEAVYATOMCOUNT\":20,\"FRACTIONCSP3\":0.5,\"NUMAROMATICRINGS\":1,\"NUMSATURATEDRINGS\":0,\"NUMAROMATICHETEROCYCLES\":0,\"NUMAROMATICCARBOCYCLES\":1,\"NUMSATURATEDHETEROCYCLES\":0,\"NUMSATURATEDCARBOCYCLES\":0,\"NUMALIPHATICRINGS\":0,\"NUMALIPHATICHETEROCYCLES\":0,\"NUMALIPHATICCARBOCYCLES\":0,\"IUPAC\":\"2-butoxyethyl 2-(2,4-dichlorophenoxy)acetate\"}"""  # noqa
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
