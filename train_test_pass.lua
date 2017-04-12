-- this file is meant to feed the corrupted 3D test data to the network and save the output in binary format
--this output is then read and visualized in matlab.

-----------------------------------------------------------------
--cad model used for paper 
--classes = {  'desk', 'bathtub', 'toilet', 'monitor', 'night_stand', 'table', 'sofa', 'dresser', 'bed', 'chair'} 
--rand = {'235', '91', '271', '103', '1', '109', '43', '235', '193', '145'}
--slicing 
--input to be given ---------------------------------------------
obj_class = 'chair'
noise_type = 'rand' -- rand means random noise..Dist means slicing noise
noise_level = '30'  -- % of distortion..{10,20,30} for slicing and {10,30,50} for random noise
i = 145 		-- this is the idx of test set, a particular cad model of the class
---------------------------------------------------------------------------------------
---------------------------------------------------------------------------------
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'nn'
require 'cunn'
matio = require 'matio'
featuresOut = 27000
model = torch.load('mul-class-models/log_exp_64_256_64_.5_.1_10class_r_nll_new/'..'model.net')
LABEL = {}
criterion =  nn.MSECriterion()
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

testData.data = matio.load('Data/'..noise_type..'_'..obj_class..'_te.mat', 'te_distorted_'..noise_level)
tesize = testData.data:size()[1]
--trainData.data = matio.load('bed_tr.mat', 'tr_data')
LABEL = matio.load('Data/'..obj_class..'_te.mat', 'te_data')
testData.labels = torch.reshape(LABEL,tesize ,featuresOut)

epoch_err = 0
--print(tesize)
model:evaluate()
model:remove(1) --what happens when you dont remove it?
print(model)
--for i = 1,testData:size(),1 do

local err = 0
inputs = torch.Tensor(1,1,30,30,30) 
inputs = inputs:cuda()
input = testData.data[i]
input = input:cuda()
inputs[1] = input
target = testData.labels[i] 
outputs = model:forward(inputs)
outputs = outputs:float()
target = target:float()
err = criterion:forward(outputs, target)
epoch_err = err +epoch_err
print(err)

--end
-- to save and visualise
outputs = torch.reshape(outputs,30,30,30)
outputs = torch.squeeze(outputs)
dims = outputs:nDimension()


if dims > 1  then
    for i=1,math.floor(dims/2) do
      outputs=outputs:transpose(i, dims-i+1)
    end
    outputs = outputs:contiguous()
end

file = torch.DiskFile('recons/mat-files-paper-'..noise_type..'/'..obj_class..'/'..obj_class..'_'..noise_type..'_'..noise_level..'_'..i..'.asc', 'w')
file:writeObject(outputs)
file:close()

