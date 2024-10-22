import h5py
import numpy as np

### index of cosmology from 0 to 128
ic    = 0
### index of snapshot file from 0 to 11 --> z [3.0, 0.0]
isnap = 0
### mesh size
Nmesh = 1536
### main directory
fdir  = '/public/home/chenzhao/csst/simulation/'
fbase = fdir + 'c%04d/output/mesh_%03d/mesh_%03d_CIC_Nmesh%d'%(ic, isnap, isnap, Nmesh)

def list2slice(l):
    out = (slice(l[0], l[1], l[2]),
           slice(l[3], l[4], l[5]),
           slice(l[6], l[7], l[8]))
    return out

##### Basic information
with h5py.File(fbase+'.0.hdf5', 'r') as ff:
    boxsize = ff['Field'].attrs['BoxSize']
    nfiles = ff['Field'].attrs['NumFiles']

print(nfiles)
mesh = np.zeros((1536,1536,1536), dtype=np.float32)
for i in range(32):
    f = h5py.File(fbase+'.%d.hdf5'%i, 'r')
    slices = list2slice(f['Field'].attrs['slices'])
    mesh[slices] = f['Field'][...]
    f.close()