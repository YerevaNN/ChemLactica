from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from rd_pdbqt import MolToPDBQTBlock
from typing import List, Union

# below parameters taken from 
# https://github.com/schwallergroup/augmented_memory/blob/6170fa4181c0bc5b7523e49bdc5bfd2b60f2a6a9/beam_enumeration_reproduce_experiments/drug-discovery-experiments/drd2/docking.json # noqa
NUM_POSES = 1
VINA_SEED = 42
MAXIMUM_ITERATIONS = 600
CENTERS = [9.93, 5.85, -9.58]
SIZES = [15, 15, 15]
VINA_EXHAUSTIVENESS = 8  # please see https://github.com/MolecularAI/DockStream/blob/c62e6abd919b5b54d144f5f792d40663c9a43a5b/dockstream/utils/enums/AutodockVina_enums.py#L74 # noqa


def get_vina_score(smiles_to_score: Union[str, List], vina_obj: Vina, receptor):
    scores = []

    if isinstance(smiles_to_score, str):
        smiles_to_score = [smiles_to_score]
    vina_obj.set_receptor(receptor)
    vina_obj.compute_vina_maps(center=CENTERS, box_size=SIZES)

    for smiles in smiles_to_score:
        try:
            ligand_mol_pdbqt = smiles_to_reasonable_conformer_pdbqt(smiles)
            vina_obj.set_ligand_from_string(ligand_mol_pdbqt)
            energies = vina_obj.dock(
                exhaustiveness=VINA_EXHAUSTIVENESS, n_poses=NUM_POSES
            )
            energies = vina_obj.optimize()
            result = energies[0]  # get lowest energy binding mode
        except Exception as e:
            result = None
            print(e)
        scores.append(result)
    return scores


def smiles_to_reasonable_conformer_pdbqt(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol, maxIters=MAXIMUM_ITERATIONS)
    pdbqt_string = MolToPDBQTBlock(mol)
    return pdbqt_string


def main():
    smiles = "CCO"
    v = Vina(sf_name="vina", seed=VINA_SEED, verbosity=2)
    score = get_vina_score(smiles, v, "/mnt/sxtn2/chem/vina/6cm4-grid.pdbqt")
    print(score)


if __name__ == "__main__":
    main()
