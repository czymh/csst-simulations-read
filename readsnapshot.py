import h5py
import numpy as np

### index of cosmology from 0 to 128
ic    = 0
### index of snapshot file from 0 to 11 --> z [3.0, 0.0]
isnap = 0
### main directory
fdir  = '/public/home/chenzhao/csst/simulation/'
fbase = fdir + 'c%04d/output/snapdir_%03d/snapshot_%03d'%(ic, isnap, isnap)
##### Basic information
with h5py.File(fbase + '.0.hdf5', 'r') as ff:
    print(ff.keys())
    print(ff['Header'].attrs.keys())
    NumFiles = ff['Header'].attrs['NumFilesPerSnapshot']
    NumPartTot = ff['Header'].attrs['NumPart_Total'][1] ## 1 means Dark matter
    print(NumFiles, NumPartTot)
    print(ff['PartType1'].keys())
    print(ff['PartType1/Coordinates'].attrs.keys())

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



