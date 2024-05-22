
def load_qm9_slimes() -> list:
    path = 'mypy/data/molecule_smiles.txt'
    with open(path, 'r') as f:
        lines = f.readlines()

        out = [line.strip() for line in lines]
    
    return out