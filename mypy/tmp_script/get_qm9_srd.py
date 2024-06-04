import tarfile
from .get_split import get_splits
import torch
import numpy as np
from rdkit import Chem
from mypy.utils.molecule_transform import SRD
N = 29
D = 28

splits = get_splits()

atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

def convert_data(type: str): 
    idlist = splits[type].tolist()

    tar_path = 'qm9/temp/qm9/dsgdb9nsd.xyz.tar.bz2'
    tardata = tarfile.open(tar_path, 'r')
    files = tardata.getmembers()

    out_dict = {'num_atoms': [], 'positions': [], 'one_hot': [], 'dim_mask': [], 'charges': []}
    for file in files:
        filename = file.name
        name = filename.split('.')[0]
        id = int(name.split('_')[-1])
        with tardata.extractfile(file) as openfile:
            lines = openfile.readlines() 
            n = int(lines[0])
            smiles= lines[n+3].split()[0].decode('utf-8')

            if id in idlist:
                out_dict['num_atoms'].append(n)
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                adj = Chem.GetAdjacencyMatrix(mol)

                positions = torch.zeros(N, D)
                tp = SRD(torch.tensor(adj, dtype=torch.float))
                positions[:tp.shape[0], :tp.shape[1]] = tp

                out_dict['positions'].append(positions.tolist())

                rank = tp.shape[1]
                dim_mask = torch.zeros(D)
                dim_mask[0:rank] = 1
                dim_mask.unsqueeze(0)
                out_dict['dim_mask'].append(dim_mask.tolist())

                one_hot = torch.zeros(N, 5)
                charges = torch.zeros(N)
                atoms = mol.GetAtoms()
                for i, atom in enumerate(atoms):
                    charges[i] = charge_dict[atom.GetSymbol()]
                    atom_type = atom_encoder[atom.GetSymbol()]
                    one_hot[i, atom_type] = 1
                out_dict['one_hot'].append(one_hot.tolist())
                out_dict['charges'].append(charges.tolist())

                # verify
                mask = torch.zeros(N, N)
                mask[0:n, 0:rank] = 1
                posisions = positions * mask
                tmp = torch.zeros_like(posisions)
                tmp[0:n, 0:n] = - torch.tensor(adj, dtype=torch.float) + torch.diag(torch.sum(torch.tensor(adj, dtype=torch.float), dim=1))
                # print(torch.dist(posisions @ posisions.t(), tmp), posisions @ posisions.t(), tmp)
                assert(torch.dist(posisions @ posisions.t(), tmp) < 1e-4)
                # x = input()
                
                
    out_dict = {key: np.array(val, dtype=np.float32) for key, val in out_dict.items()}
    np.savez_compressed(f'qm9/temp/qm9_srd/{type}.npz', **out_dict)
                    

convert_data('valid')
convert_data('test')
convert_data('train')





