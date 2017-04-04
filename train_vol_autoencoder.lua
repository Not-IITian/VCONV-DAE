
--this scripts trains a volumetric auto-encoder for 10 classth tra
--with pure SGD. default training parameters are already entered.
require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'cutorch'
require 'cunn'
require 'pl'
require 'paths'

local matio = require 'matio'
----------------------------------------------------------------------
-- parse command-line options
local opt = lapp[[
   -s,--save          (default "mul-class/AE_6912_.1_10class_r_dummy/")      subdirectory to save logs
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.1)        learning rate, for SGD only
   -b,--batchSize     (default 1)          batch size
   -m,--momentum      (default 0.9)           momentum, for SGD only
]]
-- fix seed
torch.manualSeed(1234)
torch.setdefaulttensortype('torch.FloatTensor')


--define the voxel/input resolution
local inD = 30
local featuresOut = inD * inD* inD
local cube_size = inD

-- define the data-struct for stroing training and test data
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

-- load the input data
trainData.data = matio.load('Data/mul-class/mul-class_tr_10_reduced.mat', 'tr_data_10_class_r')
trsize = trainData.data:size()[1]
print ('no of training exmaple ='..trsize)
trainData.labels = torch.reshape(trainData.data,trsize ,featuresOut)
testData.data = matio.load('Data/mul-class/mul-class_te_10_reduced.mat', 'te_data_10_class_r')
tesize = testData.data:size()[1]
print( 'no of testing exmaple ='..tesize)	
testData.labels = torch.reshape(testData.data,tesize ,featuresOut)

-- define the path for log files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

--define the model parameters
desc_dims = {6912,6912} --dims of desc
fSize = {1,64,256,256,64,1} -- no of feature maps at each layer
filtsize = {9,4,5,6} --size of filters in conv-deconv layers
local dT = {2,3} --stride for deconv
local kT= 3  --upsampling (local outD = (5-1) * dT  + kT)

dropout_p = .5
model= nn.Sequential()
 model:add(nn.Dropout(dropout_p))
model:add(nn.VolumetricConvolution(fSize[1], fSize[2], filtsize[1], filtsize[1], filtsize[1], 3, 3, 3)) -- (30 - 9 + 3)/3 = 8
model:add(nn.ReLU(true))
--features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 26
model:add(nn.VolumetricConvolution(fSize[2], fSize[3], filtsize[2], filtsize[2], filtsize[2], 2, 2, 2)) -- 3^3
model:add(nn.ReLU(true))
model:add(nn.Reshape(desc_dims[1]))	
model:add(nn.Linear(desc_dims[1],desc_dims[2]))		
model:add(nn.ReLU(true))
model:add(nn.Dropout(dropout_p))
model:add(nn.Reshape(fSize[4],3,3,3 ))
--Deconvolutional layers
model:add(nn.VolumetricFullConvolution(fSize[4], fSize[5], filtsize[3], filtsize[3], filtsize[3], dT[1], dT[1], dT[1] ))
model:add(nn.ReLU(true))
model:add(nn.VolumetricFullConvolution(fSize[5], fSize[6], filtsize[4], filtsize[4], filtsize[4], dT[2], dT[2], dT[2] ))
model:add(nn.Reshape(featuresOut))
model:add(nn.Sigmoid())	
print(model)
----------------------------------------------------
-- loss function: negative log-likelihood
criterion = nn.BCECriterion()
model:cuda()
criterion:cuda()
----------------------------------------------------------------------
print('loading params from the NN')
parameters,gradParameters = model:getParameters()

--define the optimizer 
   optimState = {
      learningRate = opt.learningRate,
      --weightDecay = opt.weightDecay,
       momentum = opt.momentum,
      learningRateDecay = 5e-7        
   }
   optimMethod = optim.sgd

-- training function
function train(dataset)
   -- epoch tracker
    model:training()
   epoch = epoch or 1
   -- local vars
   local time = sys.clock()
   local shuffle = torch.randperm(trsize)

print '==> defining some tools'
   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   for t = 1,dataset.data:size()[1],opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,cube_size ,cube_size , cube_size ) 
      local targets = torch.Tensor(opt.batchSize, cube_size*cube_size*cube_size)
      inputs = inputs:cuda()
      targets =  targets:cuda()
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset.data:size()[1]) do
         -- load new sample
   
            local input = dataset.data[i]
            local target = dataset.labels[i]
            input = input:cuda()
            target =  target:cuda()
            --target = target:squeeze()
            inputs[k] = input
            targets[k] = target
             k = k + 1
      end
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         --collectgarbage()
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end
         -- reset gradients
         gradParameters:zero()
         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)        
         -- return f and df/dX
         return f,gradParameters
      end
      -- optimize on current mini-batch
         -- Perform SGD step:              
         optimMethod(feval, parameters, optimState)
         -- disp progress
         --xlua.progress(t, dataset.data:size()[1])   
   end  
   -- time taken
   time = sys.clock() - time
   time = time / dataset.data:size()[1]
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
	if  epoch%5== 0 then
   	print('<trainer> saving network to '..filename)
   	torch.save(filename, model)
	end
   -- next epoch
   epoch = epoch + 1
end
-- test function
if 0 then

function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1

      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end
      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end
   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

end

end
----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   train(trainData)
   --test(testData)
   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      --testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      --testLogger:plot()
   end
end

    
