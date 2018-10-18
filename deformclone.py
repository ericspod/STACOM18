# Copyright (c) 2018 Eric Kerfoot, see LICENSE file
'''
This script will read in the original cardiac-dig.mat file and produce a .npz file containing multiple clones of the 
original data with the spatial deformation applied. Since the augment is way too slow to use in online during training
this is needed to statically generate variations of the original data.
'''

import multiprocessing as mp
from multiprocessing import sharedctypes
import numpy as np

from scipy.interpolate import interp2d 
from scipy.ndimage import geometric_transform
from scipy.io import loadmat


def mapping(coords,interx,intery):
    '''Map the coordinates from the original image to the deformed one by adding values from the given interp2d objects.'''
    y,x=coords[:2]
    dx=interx(x,y)
    dy=intery(x,y)

    return (y+dy,x+dx)+tuple(coords[2:])


def deformBothAugment(img,mask,defrange=5):
    '''
    Deform the central parts of the image/mask pair using interpolated randomized deformation. This is a particularly
    slow implementation of this concept, however it produces good results and other versions are still too slow to use
    online during training anyway.
    '''
    grid=np.zeros((4,4,2),int)
    y=np.linspace(0,img.shape[0],grid.shape[0])
    x=np.linspace(0,img.shape[1],grid.shape[1])
    
    grid[1:3,1:3,:]=2*defrange*np.random.random_sample((2,2,2))-defrange
        
    interx=interp2d(x,y,grid[...,0],'linear')
    intery=interp2d(x,y,grid[...,1],'linear')
    xargs=(interx,intery)
        
    return geometric_transform(img,mapping,extra_arguments=xargs),geometric_transform(mask,mapping,extra_arguments=xargs)


def toShared(arr):
    '''Convert the given Numpy array to a shared ctypes object.'''
    carr=np.ctypeslib.as_ctypes(arr)
    return sharedctypes.RawArray(carr._type_, carr)


def fromShared(arr):
    '''Map the given ctypes object to a Numpy array, this is expected to be a shared object from the parent.'''
    return np.ctypeslib.as_array(arr)


data=loadmat('./cardiac-dig.mat')
    
images_LV=data['images_LV'].astype(np.float32)
endo_LV=data['endo_LV']
epi_LV=data['epi_LV']

numClones=20

output_LV=np.zeros(tuple(images_LV.shape)+(numClones+1,),dtype=images_LV.dtype)
endoout_LV=np.zeros(tuple(endo_LV.shape)+(numClones+1,),dtype=endo_LV.dtype)
epiout_LV=np.zeros(tuple(epi_LV.shape)+(numClones+1,),dtype=epi_LV.dtype)

output_LV[...,0]=images_LV
endoout_LV[...,0]=endo_LV
epiout_LV[...,0]=epi_LV

output_LV=toShared(output_LV)
endoout_LV=toShared(endoout_LV)
epiout_LV=toShared(epiout_LV)


def generate(start,end):
    '''
    This generator is executed in parallel on subprocesses, filling data into the shared arrays created above. This
    relies on fork() semantics so will not work in Windows.
    '''
    loutput_LV=fromShared(output_LV)
    lendoout_LV=fromShared(endoout_LV)
    lepiout_LV=fromShared(epiout_LV)
    
    for i in range(start,end):
        print(i,start,end)
        
        im0=images_LV[...,i]
        endo=endo_LV[...,i]
        epi=epi_LV[...,i]
        mask0=np.dstack([endo,epi])
    
        for c in range(numClones):
            im,mask=deformBothAugment(im0,mask0,5)
            loutput_LV[...,i,c+1]=im
            lendoout_LV[...,i,c+1]=mask[...,0]
            lepiout_LV[...,i,c+1]=mask[...,1]


if __name__=='__main__':
    ranges=[(a[0],a[-1]+1) for a in np.array_split(np.arange(images_LV.shape[2]),mp.cpu_count())] # list of start,end indices
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(generate,ranges)
    
    np.savez_compressed('./cardiac-dig_cloned_%i.npz'%numClones,numClones=numClones,
                        images_LV=fromShared(output_LV),
                        endo_LV=fromShared(endoout_LV),
                        epi_LV=fromShared(epiout_LV))
