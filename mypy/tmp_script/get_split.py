from qm9.data.prepare.qm9 import gen_splits_gdb9

def get_splits():
    gdb9dir = 'qm9/temp/qm9'
    splits = gen_splits_gdb9(gdb9dir, cleanup=True)

    return splits