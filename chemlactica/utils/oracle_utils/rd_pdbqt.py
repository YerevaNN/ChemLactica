from __future__ import absolute_import, print_function
from math import isnan, isinf

from rdkit import Chem
from rdkit.Chem import AllChem

METALS = (
    3,
    4,
    11,
    12,
    13,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
)


def PDBQTAtomLines(mol, donors, acceptors):
    """Create a list with PDBQT atom lines for each atom in molecule. Donors
    and acceptors are given as a list of atom indices.
    """

    atom_lines = [
        line.replace("HETATM", "ATOM  ")
        for line in Chem.MolToPDBBlock(mol).split("\n")
        if line.startswith("HETATM") or line.startswith("ATOM")
    ]

    pdbqt_lines = []
    for idx, atom in enumerate(mol.GetAtoms()):
        pdbqt_line = atom_lines[idx][:56]

        pdbqt_line += "0.00  0.00    "  # append empty vdW and ele
        # Get charge
        charge = 0.0
        fields = ["_MMFF94Charge", "_GasteigerCharge", "_TriposPartialCharge"]
        for f in fields:
            if atom.HasProp(f):
                charge = atom.GetDoubleProp(f)
                break
        # FIXME: this should not happen, blame RDKit
        if isnan(charge) or isinf(charge):
            charge = 0.0
        pdbqt_line += ("%.3f" % charge).rjust(6)

        # Get atom type
        pdbqt_line += " "
        atomicnum = atom.GetAtomicNum()
        if atomicnum == 6 and atom.GetIsAromatic():
            pdbqt_line += "A"
        elif atomicnum == 7 and idx in acceptors:
            pdbqt_line += "NA"
        elif atomicnum == 8 and idx in acceptors:
            pdbqt_line += "OA"
        elif atomicnum == 1 and atom.GetNeighbors()[0].GetIdx() in donors:
            pdbqt_line += "HD"
        else:
            pdbqt_line += atom.GetSymbol()
        pdbqt_lines.append(pdbqt_line)
    return pdbqt_lines


def MolToPDBQTBlock(mol, flexible=True, addHs=True, computeCharges=True):
    """Write RDKit Molecule to a PDBQT block

    Parameters
    ----------
        mol: rdkit.Chem.rdchem.Mol
            Molecule with a protein ligand complex
        flexible: bool (default=True)
            Should the molecule encode torsions. Ligands should be flexible,
            proteins in turn can be rigid.
        addHs: bool (default=False)
            The PDBQT format requires at least polar Hs on donors. By default Hs
            are added.
        computeCharges: bool (default=False)
            Should the partial charges be automatically computed. If the Hs are
            added the charges must and will be recomputed. If there are no
            partial charge information, they are set to 0.0.

    Returns
    -------
        block: str
            String wit PDBQT encoded molecule
    """
    # make a copy of molecule
    mol = Chem.Mol(mol)

    # if flexible molecule contains multiple fragments write them separately
    if flexible and len(Chem.GetMolFrags(mol)) > 1:
        return "".join(
            MolToPDBQTBlock(
                frag, flexible=flexible, addHs=addHs, computeCharges=computeCharges
            )
            for frag in Chem.GetMolFrags(mol, asMols=True)
        )

    # Identify donors and acceptors for atom typing
    # Acceptors
    patt = Chem.MolFromSmarts(
        "[$([O;H1;v2]),"
        "$([O;H0;v2;!$(O=N-*),"
        "$([O;-;!$(*-N=O)]),"
        "$([o;+0])]),"
        "$([n;+0;!X3;!$([n;H1](cc)cc),"
        "$([$([N;H0]#[C&v4])]),"
        "$([N&v3;H0;$(Nc)])]),"
        "$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])]"
    )
    acceptors = list(
        map(lambda x: x[0], mol.GetSubstructMatches(patt, maxMatches=mol.GetNumAtoms()))
    )
    # Donors
    patt = Chem.MolFromSmarts(
        "[$([N&!H0&v3,N&!H0&+1&v4,n&H1&+0,$([$([Nv3](-C)(-C)-C)]),"
        "$([$(n[n;H1]),"
        "$(nc[n;H1])])]),"
        # Guanidine can be tautormeic - e.g. Arginine
        "$([NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])),"
        "$([O,S;H1;+0])]"
    )
    donors = list(
        map(lambda x: x[0], mol.GetSubstructMatches(patt, maxMatches=mol.GetNumAtoms()))
    )
    if addHs:
        mol = Chem.AddHs(
            mol,
            addCoords=True,
            onlyOnAtoms=donors,
        )
    if addHs or computeCharges:
        AllChem.ComputeGasteigerCharges(mol)

    atom_lines = PDBQTAtomLines(mol, donors, acceptors)
    assert len(atom_lines) == mol.GetNumAtoms()

    pdbqt_lines = []

    # vina scores
    if (
        mol.HasProp("vina_affinity")
        and mol.HasProp("vina_rmsd_lb")
        and mol.HasProp("vina_rmsd_lb")
    ):
        pdbqt_lines.append(
            "REMARK VINA RESULT:  "
            + ("%.1f" % float(mol.GetProp("vina_affinity"))).rjust(8)
            + ("%.3f" % float(mol.GetProp("vina_rmsd_lb"))).rjust(11)
            + ("%.3f" % float(mol.GetProp("vina_rmsd_ub"))).rjust(11)
        )

    pdbqt_lines.append(
        "REMARK  Name = " + (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
    )
    if flexible:
        # Find rotatable bonds
        rot_bond = Chem.MolFromSmarts(
            "[!$(*#*)&!D1&!$(C(F)(F)F)&"
            "!$(C(Cl)(Cl)Cl)&"
            "!$(C(Br)(Br)Br)&"
            "!$(C([CH3])([CH3])[CH3])&"
            "!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&"
            "!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&"
            "!$([CD3](=[N+])-!@[#7!D1])&"
            "!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&"
            "!D1&!$(C(F)(F)F)&"
            "!$(C(Cl)(Cl)Cl)&"
            "!$(C(Br)(Br)Br)&"
            "!$(C([CH3])([CH3])[CH3])]"
        )
        bond_atoms = list(mol.GetSubstructMatches(rot_bond))
        num_torsions = len(bond_atoms)

        # Active torsions header
        pdbqt_lines.append("REMARK  %i active torsions:" % num_torsions)
        pdbqt_lines.append("REMARK  status: ('A' for Active; 'I' for Inactive)")
        for i, (a1, a2) in enumerate(bond_atoms):
            pdbqt_lines.append(
                "REMARK%5.0i  A    between atoms: _%i  and  _%i"
                % (i + 1, a1 + 1, a2 + 1)
            )

        # Fragment molecule on bonds to ge rigid fragments
        bond_ids = [mol.GetBondBetweenAtoms(a1, a2).GetIdx() for a1, a2 in bond_atoms]
        if bond_ids:
            mol_rigid_frags = Chem.FragmentOnBonds(mol, bond_ids, addDummies=False)
        else:
            mol_rigid_frags = mol
        frags = list(Chem.GetMolFrags(mol_rigid_frags))

        def weigh_frags(frag):
            """sort by the fragment size and the number of bonds (secondary)"""
            num_bonds = 0
            # bond_weight = 0
            for a1, a2 in bond_atoms:
                if a1 in frag or a2 in frag:
                    num_bonds += 1
                    # for frag2 in frags:
                    #     if a1 in frag2 or a2 in frag2:
                    #         bond_weight += len(frag2)

            # changed signs are fixing mixed sorting type (ascending/descending)
            return (
                -len(frag),
                -num_bonds,
            )  # bond_weight

        frags = sorted(frags, key=weigh_frags)

        # Start writting the lines with ROOT
        pdbqt_lines.append("ROOT")
        frag = frags.pop(0)
        for idx in frag:
            pdbqt_lines.append(atom_lines[idx])
        pdbqt_lines.append("ENDROOT")

        # Now build the tree of torsions usign DFS algorithm. Keep track of last
        # route with following variables to move down the tree and close branches
        branch_queue = []
        current_root = frag
        old_roots = [frag]

        visited_frags = []
        visited_bonds = []
        while len(frags) > len(visited_frags):
            end_branch = True
            for frag_num, frag in enumerate(frags):
                for bond_num, (a1, a2) in enumerate(bond_atoms):
                    if (
                        frag_num not in visited_frags
                        and bond_num not in visited_bonds
                        and (
                            a1 in current_root
                            and a2 in frag
                            or a2 in current_root
                            and a1 in frag
                        )
                    ):
                        # direction of bonds is important
                        if a1 in current_root:
                            bond_dir = "%i %i" % (a1 + 1, a2 + 1)
                        else:
                            bond_dir = "%i %i" % (a2 + 1, a1 + 1)
                        pdbqt_lines.append("BRANCH %s" % bond_dir)
                        for idx in frag:
                            pdbqt_lines.append(atom_lines[idx])
                        branch_queue.append("ENDBRANCH %s" % bond_dir)

                        # Overwrite current root and stash previous one in queue
                        old_roots.append(current_root)
                        current_root = frag

                        # remove used elements from stack
                        visited_frags.append(frag_num)
                        visited_bonds.append(bond_num)

                        # mark that we dont want to end branch yet
                        end_branch = False
                        break
                    else:
                        continue
                    break  # break the outer loop as well

            if end_branch:
                pdbqt_lines.append(branch_queue.pop())
                if old_roots:
                    current_root = old_roots.pop()
        # close opened branches if any is open
        while len(branch_queue):
            pdbqt_lines.append(branch_queue.pop())
        pdbqt_lines.append("TORSDOF %i" % num_torsions)
    else:
        pdbqt_lines.extend(atom_lines)

    return "\n".join(pdbqt_lines)
