import h5py
import numpy as np
import healpy as hp

### index of cosmology from 0 to 128
ic    = 0
### index of lightcone [0 or 1]
ilc   = 0
### index of cone file. each cosmology is different.
icone = 593
### main directory
fdir  = '/public/home/chenzhao/csst/simulation/'
fbase = fdir + 'c%04d/output/lightcone_%02d/conedir_%04d/conesnap_%04d'%(ic, ilc, icone, icone)
### Basic information
with h5py.File(fbase + '.0.hdf5', 'r') as ff:
    print(ff.keys())
    print(ff['Header'].attrs.keys())
    NumFiles = ff['Header'].attrs['NumFiles']
    NumPartTot = ff['Header'].attrs['NumPart_Total'][1] ## 1 means Dark matter
    print(NumFiles, NumPartTot)
    print(ff['PartType1'].keys())
    print(ff['PartType1/Coordinates'].attrs.keys())
    print(ff['HealPixHashTable'].keys())
    NumPixTot = ff['Header'].attrs['Npix_Total']
    print(NumPixTot)

### load position
pos = np.zeros((NumPartTot, 3), dtype=np.float32)
numstart = 0
for ifile in range(NumFiles):
    with h5py.File(fbase + '.%d.hdf5'%ifile, 'r') as ff:
        NumThisFile = int(ff['Header'].attrs['NumPart_ThisFile'][1])
        pos[numstart:numstart+NumThisFile] = ff['PartType1/Coordinates'][...] # unit: Mpc/h
        numstart += NumThisFile
### 'ParticleIDs', 'Velocities' are the same, Notice velocities is Gadget velocity 
### peculiar velocities = sqrt(a) * ['Velocities']

### For healpix 
nside = hp.npix2nside(NumPixTot)
print('healpix nside:', nside)
hmap = np.zeros((NumPixTot), dtype=np.float32)
numstart = 0
for ifile in range(NumFiles):
    with h5py.File(fbase + '.%d.hdf5'%ifile, 'r') as ff:
        NpixThisFile = int(ff['Header'].attrs['Npix_ThisFile'])
        hmap[numstart:numstart+NpixThisFile] = ff['HealPixHashTable/ParticleCount'][...] # unit: Mpc/h
        numstart += NpixThisFile
