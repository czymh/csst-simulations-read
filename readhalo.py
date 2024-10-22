import h5py
import numpy as np
from multiprocessing import Pool
from functools import partial

### index of cosmology from 0 to 128
ic    = 0
### index of snapshot file from 0 to 11 --> z [3.0, 0.0]
isnap = 0
### main directory
fdir  = '/public/home/chenzhao/csst/simulation/'


##### FoF halo
fbase = fdir + 'c%04d/output/groups_%03d/fof_subhalo_tab_%03d'%(ic, isnap, isnap)

##### Basic information
with h5py.File(fbase + '.0.hdf5', 'r') as ff:
    print(ff.keys())
    print(ff['Header'].attrs.keys())
    NumFiles = ff['Header'].attrs['NumFiles']
    Ngroups_Total = ff['Header'].attrs['Ngroups_Total']
    Nsubhalos_Total = ff['Header'].attrs['Nsubhalos_Total']
    print(NumFiles, Ngroups_Total, Nsubhalos_Total)
    print(ff['Group'].keys())
    print(ff['Subhalo'].keys())

## the value of readkeys can be found in the keys of ff['Group']
readkeys = ['GroupLen', 'Group_M_Crit500', 'GroupMass', 
            'Group_M_Crit200', 'Group_M_Mean200', 'Group_M_TopHat200']
### host halo mass
headers = {}
with h5py.File(fbase + '.0.hdf5', 'r') as ff:
    attrs = ff['Header'].attrs
    for ik in attrs.keys():
        headers[ik] = attrs[ik]
    group = ff['Group']
    if readkeys == None:
        readkeys = list(group.keys())
    dt_readkeys = []
    for ik in readkeys:
        if group[ik].ndim == 1:
            dt_readkeys.append((ik, group[ik].dtype))
        elif group[ik].ndimm == 2:
            dt_readkeys.append((ik, (group[ik].shape[-1], group[ik].dtype)))
        else:
            raise ValueError('add more options for ndim > 2!')
dt_readkeys = np.dtype(dt_readkeys)
nfiles = headers['NumFiles']
numtot = headers['Ngroups_Total']
boxsize = headers['BoxSize']

def ReadSingleFileH5(fname):
    ff = h5py.File(fname, 'r')
    ngroups = ff['Header'].attrs['Ngroups_ThisFile']
    data = np.zeros((ngroups), dtype=dt_readkeys)
    group = ff['Group']
    for ik in readkeys:
        data[ik] = group[ik][:]
    ff.close()
    return data
def ReadGroupG4(nprocess=1):
    with Pool(nprocess) as p:
        fnames = [fbase + '.{:d}.hdf5'.format(i) for i in range(nfiles)]
        data = np.concatenate(p.map(ReadSingleFileH5, fnames))
    if data.shape[0] != numtot:
        raise ValueError('numtot={:d} does not match readed data shape={:d}!'.format(numtot, data.shape[0]))
    print("Read data shape: ", data.shape)
    return data
data = ReadGroupG4(nprocess=4)

### subhalo is smilar to group


### Rockstar halo
fbase = fdir + 'c%04d/output/groups_all_rockstar/out_%d.list'%(ic, isnap)
### All lines for each rockstar halo catalog file,  prepare for multi-threading
nlineslist = np.loadtxt(fdir + 'c%04d/output/groups_all_rockstar/num_lines.txt'%(ic), dtype=int)
print(nlineslist)

## all columns in formation in the header
# ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z VX VY VZ JX JY JZ Spin 
# rs_klypin Mvir_all M200b M200c M500c M2500c Xoff Voff 
# spin_bullock b_to_a c_to_a A[x] A[y] A[z] 
# b_to_a(500c) c_to_a(500c) A[x](500c) A[y](500c) A[z](500c) 
# T/|U| M_pe_Behroozi M_pe_Diemer Halfmass_Radius PID

### Mvir, GroupMass, M200c, M200m, host index
usecols = [2, 7, 21, 20, 41]
def mympload(ind, fname, linearglists, usecols=usecols):
    skip, nlines = linearglists[ind]
    return np.loadtxt(fname, skiprows=skip, max_rows=nlines, usecols=usecols)

def mploadtxt(fname, nlines, nheader=17, nprocess=8):
    linesplit = np.linspace(0, nlines, nprocess+1, dtype=int)
    linearglists = [(nheader + linesplit[ii],linesplit[ii+1]-linesplit[ii]) for ii in range(nprocess)]
    func = partial(mympload, fname=fname, linearglists=linearglists)
    # print(linearglists)
    with Pool(nprocess) as p:
        data = np.vstack(p.map(func, np.arange(nprocess)))
    if data.shape[0] != nlines:
        raise ValueError('data.shape[0]={:d} != nlines={:d}!'.format(data.shape[0], nlines))
    return data

with open(fbase) as ff:
    next(ff)
    a1 = float(ff.readline().split(' ')[-1])
    z1 = 1/a1 - 1
    for il in range(3):
        next(ff)
    m_part = float(ff.readline().split(' ')[-2])
    boxsize = float(ff.readline().split(' ')[-2])
    print("m_part = {:.0f}; BoxSize = {:.0f}".format(m_part, boxsize))
    ff.close()
ncores = 8
##### Mvir, GroupMass, M200c, M200m, M200m_corr
cata_group_1 = mploadtxt(fbase, nlineslist[isnap], nprocess=ncores)
### last column is host index
### If it is -1 then it is host halo
ind_host = cata_group_1[:,-1]==-1
Npm = cata_group_1[ind_host, 1]
M_0 = cata_group_1[ind_host, 0]
M_1 = Npm*m_part
M_2 = cata_group_1[ind_host, 2]
M_3 = cata_group_1[ind_host, 3]
M_4 = M_3*(1+Npm**(-0.55))
    