
# VCONV-DAE: Deep Volumetric Shape learning without object labels
Code for reproducing experiments in https://link.springer.com/chapter/10.1007/978-3-319-49409-8_20

VCONV-DAE is a 3D volumetric denoising auto-encoder. This repository provides the data as well as code-tools
used to train the model and visualize the end results for shape completion and blending. 

# Prerequisites 
This repository has a mix of scripts written in lua and Matlab. Torch needs to be installed to train the model. For visualization purpose, Matlab is required. Note that this code is written for research purpose only. If you use this code, please cite the paper: 
Abhishek Sharma, Oliver Grau, Mario Fritz
VConv-DAE: Deep Volumetric Shape Learning Without Object Labels Inproceedings 
Geometry Meets Deep Learning Workshop at European Conference on Computer Vision (ECCV-W), 2016.

Please read the following brief description to make best use of scripts.

--train_vol_autoencoder.lua is the main file that trains the denoising auto-encoder on volumetric data.
--The Data is stored in the Data folder.


# Tables and numbers 
--mess_classifer saves the fixed length descriptor of all test set in binary format. This is later read by eval_classification script in matlab which classifies the test set using sVM (libsvm) for unsupervised performance

-- FT_trained_model.lua fine tunes the trained model for the task of classification.

--tes_overall_error.lua takes the test set for each class, noise type and amount of distortion as input and outputs the 
 completion and denoising error.


# Figures and Images
-- train_test_pass.lua can take any corrupted instance from test set, feed-forward it to trained network and saves
the output in binary format. Visualise.m matlab script then reads it and visualize the output in 3D. 

--interpolating decoder.lua is used to write the encoder and deccoder into binary file for the blending purpose. The output of this script is later fed to a mat script (view_interpolation.m) that does the blending and visualize it.



