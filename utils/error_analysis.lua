require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'nn'

matio = require 'matio'
model = torch.load('log_exp_r_d_0_32_128_32/'..'model.net')
criterion =  nn.MSECriterion()

--data = torch.rand(5,5)
--matio.save('dummy.mat',data)
trainData = {
	data = {},
	labels = {},
	size = function() return trsize end
	}
testData = {
	data = {},
	labels = {},	
	size = function() return tesize end
	}

model:evaluate()

--testData.data = matio.load('dist_te.mat', 'te_distorted_10')
testData.data = matio.load('bed_test.mat', 'te_data')
tesize = testData.data:size()[1]
--trainData.data = matio.load('bed_tr.mat', 'tr_data')
testData.labels = matio.load('bed_test.mat', 'te_data')

inputs = torch.Tensor(tesize,1,30,30,30) 
outputs = torch.Tensor(tesize,1,30,30,30) 
perfect_cubes = torch.Tensor(tesize,1,30,30,30) 

for k = 1,tesize,1 do
     input = testData.data[k]
     perfect_input = testData.labels[k]
     input = input:double()      
     perfect_input  =  perfect_input:double() 
     perfect_cubes[k] =  perfect_input                		
     inputs[k] = input       
end

outputs = model:forward(inputs)
outputs = torch.reshape(outputs,tesize,1,30,30,30)
--outputs = outputs:floats()
-- now that you have the output, estimate the error on the denoising task by only considering those voxels which were shut down at test time randomly.
err = 0
tot_noisy_voxels = 0

zero_vox_err = 0; -- 0 means we are finding the one valued voxels
for i = 1,1,1 do
	i = 1200
	noisy_cube = inputs[i]
	perfect_cube = perfect_cubes[i]

	cube_mask = torch.ne(perfect_cube,zero_vox_err) ---- this will give me a zero one tensor (1,30,30,30)indicating 1 where the object doesnt exist

	cube_zero_idx = torch.nonzero(cube_mask) -- this will give x by 4 (2 dims) matrix

	zero_vox = cube_zero_idx:size()[1]
	print(zero_vox)

		ee = cube_zero_idx
		output = outputs[i]
		
		output = output:double()
		perfect_out = perfect_cubes[i]

		err_voxel= 0
	--noisy_output = output   -- get the output of noisy voxels by proper indexing 
		
			for j = 1,zero_vox,1 do
			
		 
 				predicted_output = output[{{ee[{j,1}]},{ee[{j,2}]}, {ee[{j,3}]}, {ee[{j,4}]}}] -- output of noisy voxel one by one
			--take the diff between noisy and perfect that is all
			aa = predicted_output[1][1][1][1]
				voxel_label = perfect_out[{{ee[{j,1}]},{ee[{j,2}]}, {ee[{j,3}]}, {ee[{j,4}]}}] -- this should be always one since we only shut down voxels during noising..bug otherwise
				
				if zero_vox_err == 1 then
					err_voxel = aa^2  -- here means we are finding error of 0 voxels
				else 
					err_voxel = 1-aa
					err_voxel = err_voxel^2	

				end
				err = err +err_voxel
				bb = aa >.3 
				if bb then
					
					print(aa)
				end
				--print ('next voxel')
			end
			
			
	
end --for outer for loop on i
print(err)

--matio.save('outputs_dist3_tr.mat',outputs)



