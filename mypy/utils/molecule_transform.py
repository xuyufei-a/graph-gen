import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom, AllChem
from typing import Tuple, List

atam_decoder = ['H', 'C', 'N', 'O', 'F']
atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

def inverse_SRD(x: torch.Tensor) -> torch.Tensor:
    # x: B * N * D
    # return: B * N * N
    out = - torch.matmul(x, x.transpose(1, 2))
    tmp = torch.eye(out.size(1), dtype=torch.bool).unsqueeze(0).to(out.device)
    out *= ~tmp
    return out

def build_molecule(adjacency: torch.Tensor, atom_types: torch.Tensor) -> Chem.Mol:
    mol = Chem.RWMol()
    for atom_type in atom_types:
        id = mol.AddAtom(Chem.Atom(atam_decoder[atom_type.item()]))

    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, ]

    bonds = torch.nonzero(torch.triu(adjacency))
    for bond in bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[adjacency[bond[0], bond[1]].item()])


    atoms_to_remove = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            if len(atom.GetNeighbors()) == 0:
                atoms_to_remove.append(atom.GetIdx())

    for idx in sorted(atoms_to_remove, reverse=True):
        mol.RemoveAtom(idx)


    return mol

def SRD(adj: torch.Tensor, N: int=29, D:int=28) -> torch.Tensor:
    # adj: n * n
    
    degree = torch.sum(adj, dim=1)
    L = - adj + torch.diag(degree)

    val, vec = torch.linalg.eigh(L)
    val, vec = val[1:], vec[:, 1:]

    val = val.clamp(0, None)

    ret = vec @ torch.diag(val ** 0.5)
    assert(torch.dist(ret @ ret.t(), L) < 1e-4)
    return  ret

def legalize_valence(adjacency: torch.Tensor, atom_types: torch.Tensor, remove_h: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if remove_h:
        no_h_index = atom_types != 0
        adjacency = adjacency[no_h_index][:, no_h_index]
        atom_types = atom_types[no_h_index]    

    valence_dict = [1, 4, 3, 2, 1]
    valences = torch.zeros_like(atom_types, dtype=torch.int8)
    legal_valence = torch.zeros_like(adjacency, dtype=torch.int8)

    inf = torch.tensor(float('inf')).to(adjacency.device)
    for i in range(adjacency.size(0)):
        atom_type = atom_types[i]
        valences[i] = valence_dict[atom_type.item()]
        adjacency[i, i] = -inf
        
    n = len(adjacency)
    while True:
        index_1d, max_val = torch.argmax(adjacency).item(), torch.max(adjacency).item()
        if max_val < 0: 
            break

        r, c = index_1d // n, index_1d % n
        adjacency[r, c] -= 1
        adjacency[c, r] -= 1
        
        if valences[r] > 0 and valences[c] > 0 and legal_valence[r, c] < 3:
            valences[r] -= 1
            valences[c] -= 1
            legal_valence[r, c] += 1
            legal_valence[c, r] += 1
        else:
            adjacency[r, c] = -inf
            adjacency[c, r] = -inf

    return legal_valence, atom_types

def srd_to_smiles(srd: torch.Tensor, node_mask: torch.Tensor, atom_types: torch.Tensor, remove_h: bool=False) -> List[str]:
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
        adj, atom_type = legalize_valence(adj, atom_type, remove_h=remove_h)
        mol = build_molecule(adj, atom_type)

        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            smile = None
        else:
            mol = Chem.RemoveHs(mol)
            smile = Chem.MolToSmiles(mol)
        smiles.append(smile)

    return smiles

def smile_to_xyz(smile: str) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    result = rdDistGeom.EmbedMolecule(mol, randomSeed=42, maxAttempts=1000)
    # result = rdDistGeom.EmbedMolecule(mol)

    try:
        assert result != -1
    except:
        print(f"fail embedding on {smile}")
        return None, None

    AllChem.UFFOptimizeMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    one_hot = torch.zeros((pos.size(0), 5))
    for i, atom_type in enumerate(atom_types):
        one_hot[i, atom_encoder[atom_type]] = 1

    return pos, one_hot

#     from pyscf import gto, dft
#     from pyscf.geomopt import geometric_solver
#     import cupy as cp

#     mol = Chem.MolFromSmiles(smile)
#     mol = Chem.AddHs(mol)
#     result = rdDistGeom.EmbedMolecule(mol, randomSeed=42, maxAttempts=1000)
#     assert result != -1, "fail embedding"

#     AllChem.UFFOptimizeMolecule(mol)
#     coords = mol.GetConformer().GetPositions()
#     symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

#     pyscf_mol = gto.M(
#         atom=[(symbols[i], coords[i]) for i in range(len(symbols))],
#         # basis='6-31g'
#         basis='6-31G(2df,p)'
#     )

#     dft.numint.libxc = cp
#     mf = dft.RKS(pyscf_mol)
#     mf.xc = 'B3LYP'
#     mf.kernel()
#     new_mol = geometric_solver.optimize(mf)
#     optimized_coords = new_mol.atom_coords()

#     optimized_coords = torch.tensor(optimized_coords)
#     one_hot = torch.zeros(optimized_coords.size(0), 5)
#     for i, symbol in enumerate(symbols):
#         one_hot[i, atom_encoder[symbol]] = 1

#     return optimized_coords, one_hot
   
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