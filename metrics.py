# Copyright (c) 2018 Eric Kerfoot, see LICENSE file
'''Routines for calculating metrics from segmentations.'''

from __future__ import print_function,division
import sys
import numpy as np
from scipy.io import loadmat
from numba import jit # makes a massive runtime difference

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*') # turn off stupid zoom warning

from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import label, binary_fill_holes, binary_dilation, center_of_mass,shift, zoom, sum as ndsum
from scipy.stats import pearsonr


# segments for measuring cavity dimensions
dimsegments=(
    ((30,90),(210,270)), # dim1
    ((330,360),(0,30),(150,210)), # dim2
    ((90,150),(270,330)) # dim3
)

# segments for measuring wall thickness
wtsegments=(
    ((210,270),),
    ((150,210),),
    ((90,150),),
    ((30,90),),
    ((330,360),(0,30)),
    ((270,330),)
)

wterrs=[-0.0077164, -0.00638042, -0.0062763, -0.00617579, -0.01073589, -0.00699072]

derrs=[0.00016246754258984408, -0.0004617344629673287, -0.0008609692409942978]

numImagesPerSub=20


def isAnnulus(mask):
    '''Return True if `mask' defines an annular mask image.'''
    _,numfeatures=label(mask)
    
    if numfeatures!=1: # multiple features
        return False
    
    cavity=binary_fill_holes(mask>0).astype(mask.dtype)-mask
    
    if cavity.sum()==0: # no enclosed area
        return False
    
    _,numfeatures=label(cavity)
    
    return numfeatures==1 # exactly 1 enclosed area
    

def isolateLargestMask(mask):
    '''Label the binary images in `mask' and return an image retaining only the largest.'''
    labeled,numfeatures=label(mask) # label each separate object with a different number
    
    if numfeatures>1: # if there's more than one object in the segmentation, keep only the largest as the best guess
        sums=ndsum(mask,labeled,range(numfeatures+1)) # sum the pixels under each label
        maxfeature=np.where(sums==max(sums)) # choose the maximum sum whose index will be the label number
        mask= mask*(labeled==maxfeature) # mask out the prediction under the largest label

    return mask


def isolateCavity(mask):
    '''Returns the cavity mask from given annulus mask.'''
    assert mask.sum()>0
    cavity=binary_fill_holes(mask>0).astype(mask.dtype)-mask
    
    if cavity.sum()==0:
        return isolateCavity(closeMask(mask)) # try again with closed mask
    else:
        return cavity


def calculateConvexHull(mask):
    '''Returns a binary mask convex hull of the given binary mask.'''
    m=mask>0

    region=np.argwhere(m)
    hull=ConvexHull(region)
    de=Delaunay(region[hull.vertices])

    simplexpts=de.find_simplex(np.argwhere(m==m))
    return simplexpts.reshape(m.shape)!=-1


def closeMask(mask):
    '''Returns a mask with a 1 pixel rim added if need to ensure the mask is an annulus.'''
    if not isAnnulus(mask):
        hullmask=calculateConvexHull(mask)
        largemask=binary_dilation(hullmask,iterations=1)
        rim=largemask.astype(mask.dtype)-hullmask.astype(mask.dtype)
    else:
        rim=0
        
    return mask+rim
    

def centerCavity(cavity):
    '''Center the given cavity image.'''
    cy,cx= center_of_mass(cavity)
    return shift(cavity,(cavity.shape[0]//2-cy,cavity.shape[1]//2-cx))


def centerMask(mask):
    '''Center the given mask using the cavity's center of mass.'''
    cavity=isolateCavity(mask)
    cy,cx= center_of_mass(cavity)
    return shift(mask,(mask.shape[0]//2-cy,mask.shape[1]//2-cx))


def imageToPolarMap(inh,inw,outh,outw):
    '''
    Defines a grid of coordinates mapping pixels from an image of dimensions (inh,inw) to polar coordinates
    stored in an array of dimensions (outh,outw,2). In the output array, if a value at (r,t) is a pair of
    positive values (y,x), then the pixel in the original image at (y,x) is equivalent to the polar coord
    (r,t) where r is the distance from the image center, and t is the angle in radians clockwise from the 
    top of the image. The result is the coords of the original image projected into a polar rectangle area.
    '''
    output=np.zeros((outh,outw,2),np.int32)-1
    my=inh//2
    mx=inw//2
    #maxlen=1.0/np.sqrt(mx**2+my**2)
    pi2=1.0/(np.pi*2)
    
    for y,x in np.ndindex(inh,inw):
        py=y-my
        px=x-mx
        rho=np.sqrt(px**2+py**2)#*maxlen*(outh-1)
        theta=(np.pi-np.math.atan2(px,py))*pi2*(outw-1)
        
        assert 0<=rho<outh and 0<=theta<outw,'%f %f'%(rho,theta)
        
        output[int(rho),int(theta)]=y,x
        
    return output


def meanAbsError(x,y):
    '''Returns the mean absolute error between `x' and `y'.'''
    return np.abs(x-y).mean()


def pearson(x,y):
    '''Returns the pearson correlation coefficient between `x' and `y'.'''
    return pearsonr(x.flatten(),y.flatten())[0]


def errorRate(x,y):
    '''Returns the mean dissimilarity between `x' and `y'.'''
    return (x!=y).sum()


def calculateAreas(masks):
    '''
    Calculates the two area metrics from `masks' which must be in BHW dimensional order. Result is a 2-by-N
    matrix storing the cavity and myocardial areas respectively for each image, scaled by image dimensions.
    '''
    result=np.zeros((2,masks.shape[0]))
    ratio=masks.shape[1]*masks.shape[2]
    
    for m,mask in enumerate(masks):
        #mask=isolateLargestMask(mask)
        cavity=isolateCavity(mask)
        result[:,m]=cavity.sum()/ratio, mask.sum()/ratio
            
    return result


def calculateCavityDims(masks,widthscale=3,imgscale=10):
    '''
    Calculates the three cavity dimensions from `masks' which must be in BHW order. This assumes a RV left 
    orientation.Result is a 3-by-N matrix storing the diagonal lengths scaled by image dimensions.
    '''
    h=masks.shape[1]*imgscale
    w=masks.shape[2]*imgscale
    ratio=2.0/h
    width=360*widthscale
    
    pmap=imageToPolarMap(h,w,h,width)
    validmapvals=np.argwhere(pmap[...,0]>=0)
    result=np.zeros((3,masks.shape[0]))
    
    heights=np.zeros((width,))
    
    @jit(nopython=True)
    def calcHeights(heights,mask):
        for v in range(validmapvals.shape[0]):
            i,j=validmapvals[v]
            y,x=pmap[i,j]
            if mask[y,x]:
                heights[j]=max(heights[j],i)

    for m,mask in enumerate(masks):
        #mask=isolateLargestMask(mask)
        cavity=isolateCavity(mask)
        cavity=centerCavity(cavity)
        cavity=zoom(cavity>0,(imgscale,imgscale),order=0)

        heights[...]=0
        calcHeights(heights,cavity)

        for dim,seg in enumerate(dimsegments):
            meanheight=np.hstack([heights[s*widthscale:e*widthscale] for s,e in seg]).mean()

            result[dim,m]=meanheight*ratio
        
    return result+np.stack([derrs]*masks.shape[0],axis=1)


def calculateWallThicknesses(masks,widthscale=3,imgscale=10):
    '''
    Calculates the 6 thickness metrics for the walls in `masks' which must be in BHW dimensional order.
    This assumes a RV left orientation. Returns a 6-by-N matrix of wall thickness values.
    '''
    result=np.zeros((6,masks.shape[0]))
    h=masks.shape[1]*imgscale
    w=masks.shape[2]*imgscale
    ratio=1.0/h
    
    width=360*widthscale
    pmap=imageToPolarMap(h,w,h,width)
    validmapvals=np.argwhere(pmap[...,0]>=0)
    heights=np.zeros((width,2))
    
    @jit(nopython=True)
    def calcHeights(heights,mask):
        for v in range(validmapvals.shape[0]):
            i,j=validmapvals[v]
            y,x=pmap[i,j]
            if mask[y,x]:
                heights[j,0]=min(heights[j,0],i)
                heights[j,1]=max(heights[j,1],i)
                
                
    for m,mask in enumerate(masks):
        #mask=isolateLargestMask(mask)
        mask=closeMask(mask)
        mask=centerMask(mask)
        mask=zoom(mask>0,(imgscale,imgscale),order=0)
        
        heights[...,0]=h
        heights[...,1]=0
        
        calcHeights(heights,mask)
                
        for wt,seg in enumerate(wtsegments):
            minheight=np.hstack([heights[s*widthscale:e*widthscale,0] for s,e in seg])
            maxheight=np.hstack([heights[s*widthscale:e*widthscale,1] for s,e in seg])
            meandiff=np.mean(np.abs(maxheight-minheight))
            
            result[wt,m]=meandiff*ratio
        
    return result+np.stack([wterrs]*masks.shape[0],axis=1)


def calculatePhases(areas):
    areas=areas.reshape((areas.shape[0]//numImagesPerSub,numImagesPerSub))
    result=np.zeros(areas.shape,np.int32)
    
    for i,a in enumerate(areas):
        imin=a.argmin()+1
        imax=a.argmax()+1
        
        if imin<imax:
            result[i,imin:imax]=1
        else:
            result[i,:imax]=1
            result[i,imin:]=1
        
    return (1-result.flatten())