import h5py
import numpy as np
import healpy as hp

### index of cosmology from 0 to 128
ic    = 0
### index of massmap, each cosmology is different
imm   = 0

### main directory
fdir  = '/public/home/chenzhao/csst/simulation/'
fbase = fdir + 'c%04d/output/mapsdir_%03d/maps_%03d'%(ic, imm, imm)
### Basic information
with h5py.File(fbase + '.0.hdf5', 'r') as ff:
    print(ff.keys())
    print(ff['Header'].attrs.keys())
    NumFiles  = ff['Header'].attrs['NumFiles']
    NpixTotal = ff['Header'].attrs['NpixTotal']
    Nside     = ff['Header'].attrs['Nside']
    print(NumFiles)
    print(NpixTotal)
    print(Nside, 12*Nside*Nside)

hpmap  = np.zeros((NpixTotal), dtype=np.float32)
istart = 0
iend   = 0
for ifile in range(NumFiles):
    with h5py.File(fbase + '.%d.hdf5'%ifile, 'r') as ff:
        iend += ff['Header'].attrs['NpixLocal']
        hpmap[istart:iend] = ff['Maps/Mass'][...]
    istart = iend
print(iend)
