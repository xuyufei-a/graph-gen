import torch
from rdkit import Chem

def inverse_SRD(x: torch.Tensor) -> torch.Tensor:
    # x: B * N * D
    out = torch.matmul(x, x.transpose(1, 2)).sigmoid()

    out = (out * 4).floor().to(dtype=torch.int8)
    tmp = torch.eye(out.size(1), dtype=torch.bool).unsqueeze(0).to(out.device)
    out *= ~tmp
    out = out.clamp(0, 3)

    return out

def build_molecule(adjacency: torch.Tensor, atom_types: torch.Tensor) -> Chem.Mol:
    atam_decoder = ['H', 'C', 'N', 'O', 'F']

    mol = Chem.RWMol()
    for atom_type in atom_types:
        id = mol.AddAtom(Chem.Atom(atam_decoder[atom_type.item()]))

    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

    bonds = torch.nonzero(torch.triu(adjacency))
    print(adjacency)
    for bond in bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[adjacency[bond[0], bond[1]].item()])

    return mol

def SRD(adj: torch.Tensor, N: int=29, D:int=18) -> torch.Tensor:
    # adj: n * n
    
    degree = torch.sum(adj, dim=1)
    A = adj + torch.diag(degree)

    val, vec = torch.linalg.eig(A)

    return torch.diag(val ** 0.5) * vec