import tarfile
from get_split import get_splits


splits = get_splits()
idlist = splits['train'].tolist()

tar_path = 'qm9/temp/qm9/dsgdb9nsd.xyz.tar.bz2'
tardata = tarfile.open(tar_path, 'r')
files = tardata.getmembers()
for file in files:
    filename = file.name
    name = filename.split('.')[0]
    id = int(name.split('_')[-1])
    with tardata.extractfile(file) as openfile:
        # print(file)
        # exit()
        lines = openfile.readlines() 
        n = int(lines[0])
        SLIMES = lines[n+3].split()[0].decode('utf-8')

        if id in idlist:
            print(n, SLIMES)
