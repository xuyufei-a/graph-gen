
def load_qm9_slimes() -> list:
    path = 'mypy/data/qm9_with_h_smiles.txt'
    with open(path, 'r') as f:
        lines = f.readlines()

        out = [line.strip() for line in lines]
    
    return out