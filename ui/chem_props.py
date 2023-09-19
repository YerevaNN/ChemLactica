from enum import Enum, auto


class ChemLacticaProperty(Enum):
    SMILES = auto()
    CID = auto()
    SAS = auto()
    WEIGHT = auto()
    TPSA = auto()
    CLOGP = auto()
    QED = auto()
    NUMHDONORS = auto()
    NUMHACCEPTORS = auto()
    NUMHETEROATOMS = auto()
    NUMROTATABLEBONDS = auto()
    NOCOUNT = auto()
    NHOHCOUNT = auto()
    RINGCOUNT = auto()
    HEAVYATOMCOUNT = auto()
    FRACTIONCSP3 = auto()
    NUMAROMATICRINGS = auto()
    NUMSATURATEDRINGS = auto()
    NUMAROMATICHETEROCYCLES = auto()
    NUMAROMATICCARBOCYCLES = auto()
    NUMSATURATEDHETEROCYCLES = auto()
    NUMSATURATEDCARBOCYCLES = auto()
    NUMALIPHATICRINGS = auto()
    NUMALIPHATICHETEROCYCLES = auto()
    NUMALIPHATICCARBOCYCLES = auto()
    IUPAC = auto()

    def __str__(self):
        return self.name
