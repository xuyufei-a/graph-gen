import torch
from rdkit import Chem
from rdkit.Chem.rdchem import AtomValenceException
from utils.molecule_transform import build_molecule

class MoleculeMetrics:
    def __init__(self, dataset_smiles_list: list):
        self.dataset_smiles_list = dataset_smiles_list

    def compute_validity(self, molecules: list):
        valid = []
        for adjacency, atom_types in molecules:
            mol = build_molecule(adjacency, atom_types)

            try:
                mol = Chem.RemoveHs(mol)
            except AtomValenceException:
                smiles = None
            else:
                smiles = Chem.MolToSmiles(mol)



            if smiles is not None:
                valid.append(smiles)

        return valid, len(valid) / len(molecules) if len(molecules) > 0 else 0.0
    
    def compute_uniqueness(self, valid: list):
        return list(set(valid)), len(set(valid)) / len(valid) if len(valid) > 0 else 0.0

    def compute_novelty(self, unique: list):
        novel = []
        for smile in unique:
            if smile not in self.dataset_smiles_list:
                novel.append(smile)
        return novel, len(novel) / len(unique) if len(unique) > 0 else 0.0
    
    def evaluate(self, molecules: list):
        valid, validity = self.compute_validity(molecules)
        unique, uniqueness = self.compute_uniqueness(valid)
        novel, novelty = self.compute_novelty(unique)
        return validity, uniqueness, novelty