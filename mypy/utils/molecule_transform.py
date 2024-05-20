import torch
from rdkit import Chem

def inverse_SRD(x: torch.Tensor) -> torch.Tensor:
    # x: B * N * D

    out = torch.matmul(x, x.transpose(1, 2)).fill_diagonal_(0).sigmoid_()
    out = (out * 4).floor() 
    return out

def build_molecule(adjacency: torch.Tensor, atom_types: torch.Tensor) -> Chem.Mol:
    atam_decoder = ['H', 'C', 'N', 'O', 'F']

    mol = Chem.RWMol()
    for atom_type in atom_types:
        mol.AddAtom(Chem.Atom(atam_decoder[atom_type.item()]))

    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

    bonds = torch.nonzero(torch.triu(adjacency))
    for bond in bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[adjacency[bond[0], bond[1]].item()])

    return mol