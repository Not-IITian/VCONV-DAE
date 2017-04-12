
-- this script takes as input a pre-trained autoencoder model and fine tunes it 
--for classification
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
   -s,--save          (default "mul-class/AE_6912_fc_6912_512_10clas_.0001/")      subdirectory to save logs
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.0001)        learning rate, for SGD only
   -b,--batchSize     (default 50)          batch size
   -m,--momentum      (default 0.9)           momentum, for SGD only
]]
-- fix seed
torch.manualSeed(1234)
torch.setdefaulttensortype('torch.FloatTensor')
-- threads
tot_epochs = 500
dropout_p = .5
-- load the model you want to fine tune for classifcation
model = torch.load('mul-class/AE_6912_.1_10class_r/'..'model.net')
no_outputs = 10

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

local inD = 30
local featuresOut = inD * inD* inD
local cube_size = 30
trainData.data = matio.load('Data/mul-class/mul-class_tr_10_reduced.mat', 'tr_data_10_class_r')
trsize = trainData.data:size()[1]
print (trsize)
trainData.labels = matio.load('Data/mul-class/mul-class_tr_10_reduced.mat', 'tr_labels_r') 
testData.data = matio.load('Data/mul-class/mul-class_te_10_reduced.mat', 'te_data_10_class_r')
tesize = testData.data:size()[1]
testData.labels = matio.load('Data/mul-class/mul-class_te_10_reduced.mat', 'te_labels_r')
print( tesize)	

classes = {}
for i=1,no_outputs do
	classes[i]=i	
end

testData.labels = torch.reshape(testData.data,tesize ,featuresOut)
  --dims of cube
desc_dims = {6912,6912}
-- features size
fSize = {1,64,256,256,64,1} -- cos now we have a fc layer
--fSize = {1,128,256,128,1}--
-- hidden units, filter sizes (for ConvNet only):
filtsize = {9,4,5,6}
--calculation for deconvolution layers
local dT = {2,3} --stride for deconv
local kT= 3  --upsampling
--local outD = (5-1) * dT  + kT
	--model = torch.load('mul-class/log_exp_64_256_64_.5_.001_40class_r_10_classifier/model.net')
	--noutputs = opt.no_outputs	
	model:remove(15)  --remove sigmoid as well
	model:remove(14)
	model:remove(13)
	model:remove(12)
	model:remove(11)
	model:remove(10)  --remove sigmoid as well
	
	--model:add(nn.Dropout(dropout_p))
	model:add(nn.Linear(6912,512))	
	model:add(nn.ReLU(true))
	model:add(nn.Dropout(dropout_p))
	model:add(nn.Linear(512,no_outputs))
	model:add(nn.LogSoftMax())
print(model)
----------------------------------------------------
-- loss function: negative log-likelihood
criterion = nn.ClassNLLCriterion()  
model:cuda()
criterion:cuda()
----------------------------------------------------------------------
-- define training and testing functions
-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
parameters,gradParameters = model:getParameters()
print('loading params from the NN')
------------------
   optimState = {
      learningRate = opt.learningRate,
      --weightDecay = opt.weightDecay,
       momentum = opt.momentum,
      learningRateDecay = 5e-7       
   }
   optimMethod = optim.sgd
-- training function
confusion = optim.ConfusionMatrix(classes)
-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

parameters,gradParameters = model:getParameters()
print('loading params from the NN')
------------------
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
   print '==> defining some tools'
   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   for t = 1,dataset.data:size()[1],opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,cube_size ,cube_size , cube_size ) 
      targets = torch.Tensor(opt.batchSize)	
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
         -- update confusion
         confusion:batchAdd(outputs, targets)
         -- return f and df/dX
         return f,gradParameters
      end
         -- Perform SGD step:      
         optimMethod(feval, parameters, optimState)
         -- disp progress
         xlua.progress(t, dataset.data:size()[1])   
   end
   -- time taken
   time = sys.clock() - time
   time = time / dataset.data:size()[1]
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
   -- print confusion matrix
   --confusion:updateValids()
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)
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
for t = 1,tot_epochs do
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

    
