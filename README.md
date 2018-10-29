
# Vconv-dae: Deep Volumetric Shape learning without object labels
Code for reproducing experiments in https://link.springer.com/chapter/10.1007/978-3-319-49409-8_20
VCONV-DAE is a 3D volumetric denoising auto-encoder. This repository provides the data as well as code-tools
used to train the model and visualize the end results for shape completion and blending. If you use this code, please cite the paper: 

VConv-DAE: Deep Volumetric Shape Learning Without Object Labels 

 at European Conference on Computer Vision (ECCVW), 2016.

For any question regarding this code, please send an email to kein.iitian@gmail.com. Note that various recent work consider Vconv-dae as a baseline and thus, may have the same implementation available in different frameworks. 
# Prerequisites 
This repository has a mix of scripts written in lua and Matlab. Torch needs to be installed to train the model. For visualization purpose, Matlab is required. Note that this code is written for research purpose only. Please read the following brief description to make best use of scripts.

--train_vol_autoencoder.lua is the main file that trains the denoising auto-encoder on volumetric data.
--The Data is stored in the Data folder.


# Shape Classification and Completion Quantitative Results 
--mess_classifer saves the fixed length descriptor of all test set in binary format. This is later read by eval_classification script in matlab which classifies the test set using sVM (libsvm) for unsupervised performance

-- FT_trained_model.lua fine tunes the trained model for the task of classification.

--tes_overall_error.lua takes the test set for each class, noise type and amount of distortion as input and outputs the 
 completion and denoising error.


# Shape Completion Qualitative results 
-- train_test_pass.lua can take any corrupted instance from test set, feed-forward it to trained network and saves
the output in binary format. Visualise.m matlab script then reads it and visualize the output in 3D. 

--interpolating decoder.lua is used to write the encoder and deccoder into binary file for the blending purpose. The output of this script is later fed to a mat script (view_interpolation.m) that does the blending and visualize it.



