--this script reproduces the shape completion numbers of our papers.
--For each class in ModelNet10, it takes input as a mat file of all its instances
--and outputs the average error 


obj_class = 'chair'
noise_type = 'rand' -- rand means random noise..Dist means slicing noise
noise_level = '30'  -- % of distortion..{10,20,30} for slicing and {10,30,50} for random noise

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'cunn'

matio = require 'matio'
--model = torch.load('mul-class/AE_6912_.1_10class_r/'..'model.net')
model = torch.load('mul-class-models/AE_6912_.1_10class_r_dummy/'..'model.net')
print(model)
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

--testData.data = matio.load('Data/rand_dresser_te.mat', 'te_distorted_50')
testData.data = matio.load('Data/'..noise_type..'_'..obj_class..'_te.mat', 'te_distorted_'..noise_level)
tesize = testData.data:size()[1]
--trainData.data = matio.load('bed_tr.mat', 'tr_data')
testData.labels = matio.load('Data/'..obj_class..'_te.mat', 'te_data')

model:evaluate()
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

inputs = inputs:cuda()
outputs = model:forward(inputs)
outputs = outputs:double()
outputs = torch.reshape(outputs,tesize,1,30,30,30)
--outputs = outputs:floats()
-- now that you have the output, estimate the error on the denoising task by only considering those voxels which were shut down at test time randomly.
err = 0
--print('done')
for i = 1,tesize,1 do
	output = outputs[i]
	bin_output = torch.gt(output, .5 )
	bin_output = bin_output:double()
	perfect_cube = perfect_cubes[i]
	noisey_voxels_tensor = torch.ne(bin_output,perfect_cube)  -- this will give me a zero one tensor (1,30,30,30)indicating 1 where the cubes are equal in value and 0 otherwise	
	noisey_voxels_idx = torch.nonzero(noisey_voxels_tensor) -- this will give x by 4 (2 dims) matrix
	dummy = torch.numel(noisey_voxels_idx)
	if dummy > 0 then
		err = err + noisey_voxels_idx:size()[1]
	        --print(err)
	else
	    print('no error in this example')
	end		
end --for outer for loop on i
--print(tesize)
aa = err/tesize
--print(err/tesize)
te_err = aa*100/13824
print('the test error is '..te_err..'%')



