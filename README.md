# STACOM18
STACOM 2018 Challenge Code

This is the code used for the submission to the LVQuant challenge and published as **Left-Ventricle Quantification Using Residual U-Net**. It relies on PyTorch 0.4.0+, Numpy, Scipy, and the included submodule **DeepLearnUtils**. When checking out this repo you may have to run the following to get the submodule:

    git submodule update --init --recursive
    
To start training, first copy the train data `cardiac-dig.mat` and test data `lvquan_test_images_30sub.mat` files into this directory. Run `deformclone.py` to generate the augmented training dataset. Training of the model is then done in `STACOMTrain.ipynb` and evaluation of metrics in `STACOMTestEval.ipynb`. 
