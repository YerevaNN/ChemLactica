"""Canonical PMO conditioning for task-agnostic and task-informed runs.

Task-informed prompts only use structured tags found in the Chemlactica
training format.  Property specifications are rebuilt for every call because
the optimizer attaches per-molecule values to them during a run.
"""

from collections import OrderedDict
from typing import Any, Callable

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors


PMO_TASKS = (
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
)

PMO_PROMPT_MODES = ("task-agnostic", "task-informed")

TARGET_SMILES = {
    "albuterol": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "amlodipine": r"Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC",
    "camphor": "CC1(C)C2CCC1(C)C(=O)C2",
    "celecoxib": "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
    "menthol": "CC(C)C1CCC(C)CC1O",
    "mestranol": "COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
    "osimertinib": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34",
    "perindopril": "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC",
    "pharmacophore": "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C",
    "ranolazine": "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2",
    "sildenafil": "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    "sitagliptin": "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F",
    "tadalafil": "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C",
    "thiothixene": "CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
    "troglitazone": "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
    "fexofenadine": "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
    "zaleplon": "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1",
}

# The exact SMARTS literal used by the pinned Valsartan oracle. It also parses
# as valid SMILES, so it can be used with the model's trained similarity tag.
VALSARTAN_SMARTS_PATTERN = "CN(C=O)Cc1ccc(c2ccccc2)cc1"


def _tag_spec(tag: str, target: str, calculator: Callable[[Any], str]):
    def infer_value(_entry):
        return target

    return {
        "start_tag": f"[{tag}]",
        "end_tag": f"[/{tag}]",
        "infer_value": infer_value,
        "calculate_value": calculator,
    }


def _similarity_spec(
    reference_smiles: str,
    desired_similarity: float,
    *,
    preserve_reference_text: bool = False,
):
    reference_mol = Chem.MolFromSmiles(reference_smiles)
    if reference_mol is None:
        raise ValueError(f"Invalid reference SMILES: {reference_smiles}")
    reference_fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        reference_mol, 2, nBits=2048
    )
    canonical_reference = Chem.MolToSmiles(reference_mol, canonical=True)
    prompt_reference = (
        reference_smiles if preserve_reference_text else canonical_reference
    )

    def infer_value(_entry):
        return f"{prompt_reference} {desired_similarity:.2f}"

    def calculate_value(entry):
        similarity = DataStructs.TanimotoSimilarity(
            reference_fingerprint, entry.fingerprint
        )
        return f"{prompt_reference} {similarity:.2f}"

    return {
        "start_tag": "[SIMILAR]",
        "end_tag": "[/SIMILAR]",
        "infer_value": infer_value,
        "calculate_value": calculate_value,
    }


def _properties(*items):
    return OrderedDict(items)


def _qed(target=0.95):
    return _tag_spec("QED", f"{target:.2f}", lambda entry: f"{QED.qed(entry.mol):.2f}")


def _clogp(target):
    return _tag_spec(
        "CLOGP", f"{target:.2f}", lambda entry: f"{Descriptors.MolLogP(entry.mol):.2f}"
    )


def _tpsa(target):
    return _tag_spec(
        "TPSA",
        f"{target:.2f}",
        lambda entry: f"{rdMolDescriptors.CalcTPSA(entry.mol):.2f}",
    )


def _formula(target):
    return _tag_spec(
        "FORMULA", target, lambda entry: rdMolDescriptors.CalcMolFormula(entry.mol)
    )


def _ring_count(target):
    return _tag_spec(
        "RINGCOUNT",
        str(target),
        lambda entry: str(rdMolDescriptors.CalcNumRings(entry.mol)),
    )


def _aromatic_ring_count(target):
    return _tag_spec(
        "NUMAROMATICRINGS",
        str(target),
        lambda entry: str(rdMolDescriptors.CalcNumAromaticRings(entry.mol)),
    )


def get_pmo_additional_properties(task_name: str, mode: str = "task-agnostic"):
    """Return optimizer property specs for a PMO task and information regime.

    ``task-agnostic`` returns no task-derived conditioning. ``task-informed``
    uses the canonical structured-prefix registry below.
    """
    if task_name not in PMO_TASKS:
        raise ValueError(f"Unknown PMO task: {task_name}")
    if mode not in PMO_PROMPT_MODES:
        raise ValueError(
            f"Unknown PMO prompt mode: {mode}. Expected one of {PMO_PROMPT_MODES}."
        )
    if mode == "task-agnostic":
        return _properties()
    return _build_task_informed_properties(task_name)


def _build_task_informed_properties(task_name: str):
    # These classifier names were not part of the models' structured vocabulary.
    if task_name in {"jnk3", "gsk3b", "drd2"}:
        return _properties()
    if task_name == "qed":
        return _properties(("qed", _qed(0.95)))
    if task_name == "albuterol_similarity":
        return _properties(
            ("sim_albuterol", _similarity_spec(TARGET_SMILES["albuterol"], 0.75))
        )
    if task_name == "mestranol_similarity":
        return _properties(
            ("sim_mestranol", _similarity_spec(TARGET_SMILES["mestranol"], 0.75))
        )
    if task_name == "celecoxib_rediscovery":
        return _properties(
            ("sim_celecoxib", _similarity_spec(TARGET_SMILES["celecoxib"], 0.99))
        )
    if task_name == "thiothixene_rediscovery":
        return _properties(
            ("sim_thiothixene", _similarity_spec(TARGET_SMILES["thiothixene"], 0.99))
        )
    if task_name == "troglitazone_rediscovery":
        return _properties(
            ("sim_troglitazone", _similarity_spec(TARGET_SMILES["troglitazone"], 0.99))
        )
    if task_name == "osimertinib_mpo":
        return _properties(
            ("sim_osimertinib", _similarity_spec(TARGET_SMILES["osimertinib"], 0.80)),
            ("tpsa", _tpsa(100.0)),
            ("clogp", _clogp(1.0)),
        )
    if task_name == "fexofenadine_mpo":
        return _properties(
            ("sim_fexofenadine", _similarity_spec(TARGET_SMILES["fexofenadine"], 0.80)),
            ("tpsa", _tpsa(90.0)),
            ("clogp", _clogp(4.0)),
        )
    if task_name == "ranolazine_mpo":
        return _properties(
            ("sim_ranolazine", _similarity_spec(TARGET_SMILES["ranolazine"], 0.70)),
            ("tpsa", _tpsa(95.0)),
            ("clogp", _clogp(7.0)),
        )
    if task_name == "perindopril_mpo":
        return _properties(
            ("sim_perindopril", _similarity_spec(TARGET_SMILES["perindopril"], 0.99)),
            ("aromatic_rings", _aromatic_ring_count(2)),
        )
    if task_name == "amlodipine_mpo":
        return _properties(
            ("sim_amlodipine", _similarity_spec(TARGET_SMILES["amlodipine"], 0.99)),
            ("ring_count", _ring_count(3)),
        )
    if task_name == "zaleplon_mpo":
        return _properties(
            ("sim_zaleplon", _similarity_spec(TARGET_SMILES["zaleplon"], 0.99)),
            ("formula", _formula("C19H17N3O2")),
        )
    if task_name == "sitagliptin_mpo":
        return _properties(
            ("clogp", _clogp(2.02)),
            ("tpsa", _tpsa(77.04)),
            ("formula", _formula("C16H15F6N5O")),
        )
    if task_name == "median1":
        return _properties(
            ("sim_camphor", _similarity_spec(TARGET_SMILES["camphor"], 0.55)),
            ("sim_menthol", _similarity_spec(TARGET_SMILES["menthol"], 0.55)),
        )
    if task_name == "median2":
        return _properties(
            ("sim_tadalafil", _similarity_spec(TARGET_SMILES["tadalafil"], 0.57)),
            ("sim_sildenafil", _similarity_spec(TARGET_SMILES["sildenafil"], 0.57)),
        )
    if task_name == "isomers_c7h8n2o2":
        return _properties(("formula", _formula("C7H8N2O2")))
    if task_name == "isomers_c9h10n2o2pf2cl":
        return _properties(("formula", _formula("C9H10N2O2PF2Cl")))
    if task_name == "deco_hop":
        return _properties(
            (
                "sim_pharmacophore",
                _similarity_spec(TARGET_SMILES["pharmacophore"], 0.85),
            )
        )
    if task_name == "scaffold_hop":
        return _properties(
            (
                "sim_pharmacophore",
                _similarity_spec(TARGET_SMILES["pharmacophore"], 0.75),
            )
        )
    if task_name == "valsartan_smarts":
        return _build_valsartan_properties()
    raise AssertionError(f"Task registry is incomplete for {task_name}")


def _build_valsartan_properties():
    """Represent the Valsartan SMARTS through the trained similarity syntax."""
    reference = Chem.MolFromSmiles(VALSARTAN_SMARTS_PATTERN)
    pattern = Chem.MolFromSmarts(VALSARTAN_SMARTS_PATTERN)
    if reference is None or pattern is None or not reference.HasSubstructMatch(pattern):
        raise AssertionError("Valsartan SMARTS must also parse as a matching SMILES")

    return _properties(
        (
            "sim_valsartan_smarts",
            _similarity_spec(
                VALSARTAN_SMARTS_PATTERN,
                0.99,
                preserve_reference_text=True,
            ),
        ),
        ("clogp", _clogp(2.0165)),
        ("tpsa", _tpsa(77.04)),
    )


def render_pmo_prefix(task_name: str, mode: str = "task-agnostic") -> str:
    """Render the exact static structured prefix used at generation time."""
    pieces = []
    for spec in get_pmo_additional_properties(task_name, mode).values():
        pieces.append(spec["start_tag"] + spec["infer_value"](None) + spec["end_tag"])
    return "".join(pieces)
