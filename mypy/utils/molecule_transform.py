import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom
from typing import Tuple, List

def inverse_SRD(x: torch.Tensor) -> torch.Tensor:
    # x: B * N * D
    # return: B * N * N
    out = torch.matmul(x, x.transpose(1, 2))
    tmp = torch.eye(out.size(1), dtype=torch.bool).unsqueeze(0).to(out.device)
    out *= ~tmp
    return out

def build_molecule(adjacency: torch.Tensor, atom_types: torch.Tensor) -> Chem.Mol:
    atam_decoder = ['H', 'C', 'N', 'O', 'F']

    mol = Chem.RWMol()
    for atom_type in atom_types:
        id = mol.AddAtom(Chem.Atom(atam_decoder[atom_type.item()]))

    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

    bonds = torch.nonzero(torch.triu(adjacency))
    for bond in bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[adjacency[bond[0], bond[1]].item()])

    return mol

def SRD(adj: torch.Tensor, N: int=29, D:int=18) -> torch.Tensor:
    # adj: n * n
    
    degree = torch.sum(adj, dim=1)
    A = adj + torch.diag(degree)

    val, vec = torch.linalg.eigh(A)

    sort_indices = torch.argsort(val, descending=True)
    val = val[sort_indices]
    vec = vec[:, sort_indices]
    val = val.clamp(0, None)

    ret = vec @ torch.diag(val ** 0.5)
    assert(torch.dist(ret @ ret.t(), A) < 1e-4)
    return  ret

def legalize_valence(adjacency: torch.Tensor, atom_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    valence_dict = [1, 4, 3, 2, 1]
    valences = torch.zeros_like(atom_types, dtype=torch.int8)
    legal_valence = torch.zeros_like(adjacency, dtype=torch.int8)

    inf = torch.tensor(float('inf')).to(adjacency.device)
    for i in range(adjacency.size(0)):
        atom_type = atom_types[i]
        valences[i] = valence_dict[atom_type.item()]
        adjacency[i, i] = -inf

    sorted_indices = torch.argsort(valences, descending=True)
    valences = valences[sorted_indices]
    atom_types = atom_types[sorted_indices]
    adjacency = adjacency[sorted_indices, :][:, sorted_indices]

    for i in range(adjacency.size(0)):
#         unit = adjacency[i].topk(valences[i]).values.sum() / valences[i]
        unit = 1

        while valences[i] > 0:
            p = torch.max(adjacency[i], dim=0).indices
            adjacency[i, p] -= unit

            if p == i or adjacency[i, p] < - 100000 * unit:
                break
            if valences[p] > 0:
                valences[i] -= 1
                valences[p] -= 1
                legal_valence[i, p] += 1
                legal_valence[p, i] += 1
            else:
                adjacency[i, p] = -inf

    return legal_valence, atom_types

def srd_to_smiles(srd: torch.Tensor, node_mask: torch.Tensor, atom_types: torch.Tensor) -> List[str]:
    # srd: B * N * D
    # node_mask: B * N
    # atom_types: B * N
    # smiles: list[str]

    adjs = inverse_SRD(srd)
    node_num = node_mask.sum(dim=1).int()
    smiles = []

    for i in range(adjs.shape[0]):
        adj = adjs[i, :node_num[i], :node_num[i]]
        atom_type = atom_types[i, :node_num[i]]
        adj, atom_type = legalize_valence(adj, atom_type)
        mol = build_molecule(adj, atom_type)

        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            smile = None
        else:
            smile = Chem.MolToSmiles(mol)
        smiles.append(smile)

    return smiles

def smile_to_xyz(smile: str) -> torch.Tensor | None:
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    rdDistGeom.EmbedMolecule(mol)
    try:
        conf = mol.GetConformer()
    except:
        pos = None
    else:
        pos = conf.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)

    return pos
   
# precated
def srd_to_xyz(srd: torch.Tensor, node_mask: torch.Tensor, atom_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # srd: B * N * D
    # node_mask: B * N
    # return: flag: B, xyz: B * N * 3

    adj = inverse_SRD(srd)
    node_num = node_mask.sum(dim=1).int()

    flag = torch.ones(srd.size(0), dtype=torch.bool, device=srd.device)
    xyz = torch.zeros(srd.shape[0:-1] + (3,), device=srd.device)

    for i in range(adj.size(0)):
        mol = build_molecule(*legalize_valence(adj[i, :node_num[i], :node_num[i]], atom_types[i, :node_num[i]]))
        smile = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        rdDistGeom.EmbedMolecule(mol)
        try:
            conf = mol.GetConformer()
        except:
            pos = None
        else:
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

        if pos is None:
            flag[i] = False
        else:
            xyz[i, 0:node_num[i]] = pos

    return flag, xyz