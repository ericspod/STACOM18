# Copyright (c) 2018 Eric Kerfoot, see LICENSE file
'''Definitions for the segmentation network.'''

import sys,os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'DeepLearnUtils')) # add DeepLearnUtils to environment
import trainutils, pytorchnet, pytorchutils, trainimagesource


def loadDataCloned(path):
    '''Returns images in HWBDC order, masks in HWBD order, phase in 1B order, and numClones as int.'''
    data=np.load(path)
    
    images_LV=data['images_LV']
    endo_LV=data['endo_LV']
    epi_LV=data['epi_LV']
    phase=data['lv_phase']
    masks=((epi_LV-endo_LV)!=0).astype(np.int32)
    numClones=int(data['numClones'])
    
    return images_LV,masks,phase,numClones 
    

class UnetMgr(pytorchutils.SegmentMgr):
    def __init__(self,params,trainData,validData,augments,savedirprefix):
        self.lossavgs=[]
        self.steplosses=[]
        self.lossavglen=50
        self.stdThreshold=7
        self.trainData=trainData
        self.validData=validData
        self.filters=params['filters']
        self.strides=params['strides']
        self.kernelsize=params['kernelsize']
        self.resunits=params['resunits']
        self.useInstanceNorm=params.get('useInstanceNorm',True)
        self.dropout=params.get('dropout',0)
        
        if trainData:
            self.src=trainimagesource.TrainImageSource(trainData[0],trainData[1],augments)
        
        net=pytorchnet.Unet2D(1,1,self.filters,self.strides,self.kernelsize,self.resunits,self.useInstanceNorm,self.dropout)
        
        super(UnetMgr,self).__init__(net,params.pop('isCuda',True),savedirprefix,**params)
        
    def train(self):
        inputfunc=self.src.getAsyncFunc(self.params['batchSize'])
        super(UnetMgr,self).train(inputfunc,self.params['trainSteps'],self.params.get('savesteps',5))
        
    def saveStep(self,step,steploss):
        losses,results=self.evaluate(self.validData,self.params['batchSize'])
        self.log('Step',step,'Mean IOU:',np.mean(results))
        
    def evalStep(self,index,steploss,results):
        masks=self.traininputs[1]
        preds=self.netoutputs[1]
        
        iou=trainutils.iouMetric(self.toNumpy(masks),self.toNumpy(preds))
        
        results.append(iou)
        
    def updateStep(self,step,steploss):
        self.steplosses.append(steploss)
        self.lossavgs.append(np.average(self.steplosses[-self.lossavglen:]))
        self.log('Loss Average:',self.lossavgs[-1])