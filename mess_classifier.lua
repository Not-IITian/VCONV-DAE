--script for mesh classification for 40 classes.
-- take a pre load netowrk and remove some layers..
-- in the end, the network should give a fixed length 1d representation for each mesh 

require 'cunn'
require 'nn'
matio = require 'matio'
model = torch.load('mul-class/AE_6912_.1_10class_r/'..'model.net')
print(model)
dummy = 1

desc_dims = 512 

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

model = model:double()
model:evaluate()
--assuming model has 9 layers..remove 6 to 9
print('removing decoder layers')
	model:remove(15)  --remove sigmoid as well
	model:remove(14)
	model:remove(13)
	model:remove(12)
	model:remove(11)	
	model:remove(10)	

print(model)

train = 0

if train == 1 then
	trainData.data = matio.load('Data/mul-class/mul-class_tr_40_reduced.mat', 'tr_data_40_class_r')
	trsize = trainData.data:size()[1]

	inputs = torch.Tensor(trsize,1,30,30,30) 
	outputs = torch.Tensor(trsize,desc_dims) 

	for k = 1,trsize,1 do
     		input = trainData.data[k]
     		input = input:double()              		
     		inputs[k] = input       
	end

-- forward each of them to network
	outputs = model:forward(inputs) 
	outputs = outputs:double()
	print('train case')
	dims = outputs:nDimension()
	if dims > 1  then
    		for i=1,math.floor(dims/2) do
      			outputs=outputs:transpose(i, dims-i+1)
    		end
    		outputs = outputs:contiguous()
	end
	file = torch.DiskFile('tr_feats_40_r_64_256_nll_ft_no_drop_512.asc', 'w')
	
	file:writeObject(outputs)
	file:close()
--for test case
else
	print('test case')
	--testData.data = matio.load('Data/mul-class/mul-class_te_40_reduced.mat', 'te_data_40_class_r')
	testData.data = matio.load('/BS/deep_3d/work/deep_3d/fcn_3D/nn-embedding/te_chair.mat', 'te_chair_data')
	tesize = testData.data:size()[1]

	inputs = torch.Tensor(tesize,1,30,30,30) 
	outputs_te = torch.Tensor(tesize,desc_dims) 

	for k = 1,tesize,1 do
     		input = testData.data[k]
     		input = input:double()              		
     		inputs[k] = input       
	end
	outputs_te = model:forward(inputs) 
	outputs_te = outputs_te:double()
	dims = outputs_te:nDimension()
	
	--inputs = inputs:cuda()
	if dims > 1  then
    		for i=1,math.floor(dims/2) do
      			outputs_te=outputs_te:transpose(i, dims-i+1)
    		end
    		outputs_te = outputs_te:contiguous()
	end

	file = torch.DiskFile('/BS/deep_3d/work/deep_3d/fcn_3D/nn-embedding/te_chair_10_64_256_nll_6912_fc_6912.asc', 'w')
	file:writeObject(outputs_te)
	file:close()
end

