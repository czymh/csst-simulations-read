import numpy as np
def xor_perpendicular(x, y):
    '''
    Computes the cross product of two 3D vectors
    '''
    # x, y 是 3D 向量
    z0 = x[1]*y[2] - x[2]*y[1]
    z1 = x[2]*y[0] - x[0]*y[2]
    z2 = x[0]*y[1] - x[1]*y[0]
    return np.array([z0, z1, z2])

# 4 means LC_TYPE_SQUAREMAP
# 0.245 astart  1.0 aend
### SquareMapZdir[0] SquareMapZdir[1]   SquareMapZdir[2]  
### 0.86779766       0.35945356         0.343104         
### SquareMapXdir[0]   SquareMapXdir[1] SquareMapXdir[2]  SquareMapAngle
### -0.38268343       0.92387953         0.              23
lc_dict = {}
lc_dict['00'] = {}
def GetLCDict(params):
    '''
    Constructs a lightcone direction dictionary from a parameter list
    only support SquareMap in Gadget4 now!
    params: SquareMapZdir[0] SquareMapZdir[1] SquareMapZdir[2] SquareMapXdir[0]   SquareMapXdir[1] SquareMapXdir[2]  SquareMapAngle
    '''
    lc_dict = {}
    lc_dict['zdir'] = np.array(params[0:3])
    lc_dict['zdir'] = lc_dict['zdir'] / np.sum(lc_dict['zdir']*lc_dict['zdir'])
    lc_dict['xdir'] = np.array(params[3:6])
    lc_dict['xdir'] = lc_dict['xdir'] / np.sum(lc_dict['xdir']*lc_dict['xdir'])
    lc_dict['ydir'] = xor_perpendicular(lc_dict['zdir'], lc_dict['xdir'])
    lc_dict['ydir'] = lc_dict['ydir'] / np.sum(lc_dict['ydir']*lc_dict['ydir'])
    lc_dict['xdir'] = xor_perpendicular(lc_dict['ydir'], lc_dict['zdir'])
    lc_dict['SquareMapAngleRad'] = np.deg2rad(params[6])
    return lc_dict

lc_dict['00'] = GetLCDict([ 0.86779766, 0.35945356, 0.343104, -0.38268343, 0.92387953, 0., 23])
lc_dict['01'] = GetLCDict([ 0.35945356, 0.86779766, 0.343104, -0.92387953, 0.38268343, 0., 23])
# print(lc_dict['00']['xdir'])
# print(lc_dict['00']['ydir'])
# print(lc_dict['00']['zdir'])
# print()
# print(lc_dict['01']['xdir'])
# print(lc_dict['01']['ydir'])
# print(lc_dict['01']['zdir'])
### x = pos * xdir
### y = pos * ydir
### z = pos * zdir
# if(z > 0 && fabs(x) < Cones[cone].SquareMapAngleRad * z && fabs(y) < Cones[cone].SquareMapAngleRad * z)
#     return true;
def is_in_cone(pos, lc_dict):
    '''
    Determines whether points are inside a lightcone
    
    pos: (N,3)
    lc_dict: output from GetLCDict
    '''
    x = np.dot(pos, lc_dict['xdir'])
    y = np.dot(pos, lc_dict['ydir'])
    z = np.dot(pos, lc_dict['zdir'])
    ind = (z > 0) * (np.abs(x) < lc_dict["SquareMapAngleRad"] * z) * (np.abs(y) < lc_dict["SquareMapAngleRad"] * z)
    return ind




