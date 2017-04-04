This doc briefly explains all the main files needed to run Vconv-DAE:

--train_vol_autoencoder.lua is the main file that trains the auto-encoder on volumetric data.
--The Data is stored in the Data folder.


--------Tables and numbers ---------------------------------------------------------------
--mess classifer saves the fixed length descriptor of all test set in binary format.
-- this is later read by eval_classification script in matlab which classifies the test
--set using sVM (libsvm) for unsupervised performance

-- FT_trained_model.lua fine tunes the trained model for the task of classification.

--tes_overall_error.lua takes the test set for each class, noise type and amount of distortion as input and outputs the 
 completion and denoising error.


-----------------------Figures----------------------------------------------------------
-- train_test_pass.lua can take any corrupted instance from test set, feed-forward it to trained network and saves
the output in binary format. Visualise.m matlab script then reads it and visualize the output in 3D. 

--interpolating decoder.lua is used to write the encoder and deccoder into binary file for the blending purpose.
--the output of this script is later fed to a mat script (view_interpolation.m) that does the blending and visualize it.


