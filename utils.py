import numpy as np

resname_3to1 = {
    "GLY": "G",
    "LEU": "L",
    "TYR": "Y",
    "SER": "S",
    "GLU": "E",
    "GLN": "Q",
    "ASP": "D",
    "ASN": "N",
    "PHE": "F",
    "ALA": "A",
    "LYS": "K",
    "ARG": "R",
    "HIS": "H",
    "CYS": "C",
    "VAL": "V",
    "PRO": "P",
    "TRP": "W",
    "ILE": "I",
    "MET": "M",
    "THR": "T",

    # Not standard
    "ASH": "D",
    "HSP": "H",
    "HSE": "H",
    "GLH": "E",
}


class Atom:
    def __init__(self):
        self.DataType = 'ATOM'  # "ATOM"/"HETATM"
        self.Name = ''  # Atom name
        self.AltLoc = ''  # Alternate location indicator.
        self.ResName = ''  # Residue name
        self.ChainId = 'A'  # Chain identifier
        self.ResSeq = 0  # Residue sequence number
        self.ResCode = ''  # Code for insertions of residues
        self.Coordin = np.array([0, 0, 0])  # (X,Y,Z) orthogonal
        self.Occup = 0.0  # Occupancy
        self.TempFac = 0.0  # Temperature factor
        self.Element = 'Xx'  # Element symbol
        self.Charge = 0.0  # Atom charge


def read_pdb(fname):
    with open(fname, "r") as file:
        molecule = dict()
        for line in file.readlines():
            data_type = line[0:6].strip()
            if data_type not in ['ATOM', 'HETATM']:
                continue
            atom = Atom()
            atom.DataType = line[0:6].strip()
            num = int(line[6:11])
            atom.Name = line[12:16].strip()
            atom.AltLoc = line[16].strip()
            atom.ResName = line[17:20].strip()
            atom.ChainId = line[21].strip()
            atom.ResSeq = int(line[22:26])
            atom.ResCode = line[26].strip()
            atom.Coordin = np.array(list(map(float, [line[30:38], line[38:46], line[46:54]])))
            atom.Occup = 0.0  # float(line[54:60])
            atom.Tempfac = 0.0  # float(line[60:66])
            atom.Element = atom.Name[0]  # line[76:78].strip()

            molecule[num] = atom

    return molecule


def read_sequence(fname):
    sequence = ""
    molecule = read_pdb(fname)
    for a in molecule.values():
        if a.Name == "CA":
            sequence += resname_3to1[a.ResName]
    return sequence


def read_coordinates(fname):
    coords = []
    molecule = read_pdb(fname)
    for atom in molecule.values():
        if atom.Name == "CA":
            coords.append(atom.Coordin)
    return coords


def read_replacements(fname):
    with open(fname) as file:
        text = file.read()
        text1 = text.split("Sets")[1].split("Pulls")[0].split()
        text2 = text.split("Pulls")[1].split("Probs")[0].split()
        text3 = text.split("Probs")[1].split("Constraints")[0]
        text4 = text.split("Constraints")[1].split()

    set = 'Set1'
    sets = {}
    for aa in text1:
        if "Set" in aa:
            set = aa
            continue
        sets[set] = sets.get(set, []) + [aa]

    pull = 'Pulls'
    pulls = {}
    for aa in text2:
        if "Pull" in aa:
            pull = aa
            continue
        pulls[pull] = pulls.get(pull, []) + [aa]

    sets_list = list(sets.keys())
    probs = {}
    for i, name in enumerate(sets_list[:-1]):
        pull_and_prob = text3.split(name)[1].split(sets_list[i+1])[0].split()
        pu = pull_and_prob[::2]
        pr = [float(i) for i in pull_and_prob[1::2]]
        probs[name] = probs.get(name, []) + [pu[:-1], pr + [float(pu[-1])]]

    pull_and_prob = text3.split(sets_list[-1])[1].split()
    pu = pull_and_prob[::2]
    pr = [float(i) for i in pull_and_prob[1::2]]
    probs[sets_list[-1]] = probs.get(sets_list[-1], []) + [pu[:-1], pr + [float(pu[-1])]]

    const_sites = []
    consts = {}
    for aa in text4:
        if "Positions" in aa:
            const_sites = []
            continue
        if "Pull" in aa:
            for i in const_sites:
                consts[int(i)] = consts.get(int(i), []) + [aa]
        else:
            const_sites.append(aa)

    return sets, pulls, probs, consts