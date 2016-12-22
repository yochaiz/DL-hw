--[[
Due to interest of time, please prepared the data before-hand into a 4D torch
ByteTensor of size 50000x3x32x32 (training) and 10000x3x32x32 (testing) 

mkdir t5
cd t5/
git clone https://github.com/soumith/cifar.torch.git
cd cifar.torch/
th Cifar10BinToTensor.lua

]]

require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('/home/yochaiz@st.technion.ac.il/cifar.torch/cifar10-train.t7')
local testset = torch.load('/home/yochaiz@st.technion.ac.il/cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

print(trainData:size())
print(trainLabels:size())

--local prefix = torch.Timer():time().real
local prefix = torch.rand(1)
print ('Prefix:[' .. prefix[1] .. ']')

--saveTensorAsGrid(trainData:narrow(1,100,36),'train_100-136.jpg') -- display the 100-136 images in dataset
--print(classes[trainLabels[100]]) -- display the 100-th image class


--  *****************************************************************
--  Let's take a look at a simple convolutional layer:
--  *****************************************************************

--[[
local img = trainData[100]:cuda()
print(img:size())

local conv = cudnn.SpatialConvolution(3, 16, 5, 5, 4, 4, 0, 0)
conv:cuda()
-- 3 input maps, 16 output maps
-- 5x5 kernels, stride 4x4, padding 0x0

print(conv)

local output = conv:forward(img)
print(output:size())
saveTensorAsGrid(output, 'convOut.jpg')

local weights = conv.weight
saveTensorAsGrid(weights, 'convWeights.jpg')
print(weights:size())
]]
--  ****************************************************************
--  Full Example - Training a ConvNet on Cifar10
--  ****************************************************************

-- Load and normalize data:

--local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
--print(#redChannel)

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end



--  ****************************************************************
--  Define our neural network
--  ****************************************************************
--[[local fullyConnectedSize = 80

local model = nn.Sequential()
model:add(cudnn.SpatialConvolution(3, 32, 5, 5)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
model:add(cudnn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
model:add(cudnn.ReLU(true))                          -- ReLU activation function
model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(cudnn.SpatialConvolution(32, 64, 3, 3))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
model:add(cudnn.ReLU(true))
model:add(nn.SpatialBatchNormalization(64))
model:add(cudnn.SpatialConvolution(64, 32, 3, 3))
model:add(nn.View(32*4*4):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
model:add(nn.Linear(32*4*4, fullyConnectedSize))             -- fully connected layer (matrix multiplication between input and weights)
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
model:add(nn.Linear(fullyConnectedSize, #classes))            -- 10 is the number of outputs of the network (in this case, 10 digits)
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classificati
]]

--local model = require 'vgg_bn_drop'

--[[local nin = require 'nin'
local model = nin()
]]

local opt = {
  num_classes = 10,
  save = 'logs',
  batchSize = 128,
  learningRate = 0.1,
  learningRateDecay = 0,
  learningRateDecayRatio = 0.2,
  weightDecay = 0.0005,
  dampening = 0,
  momentum = 0.9,
  epoch_step = "80",
  max_epoch = 300,
  model = 'nin',
  optimMethod = 'sgd',
  init_value = 10,
  depth = 4,
  shortcutType = 'A',
  nesterov = false,
  dropout = 0,
  hflip = true,
  randomcrop = 4,
  imageSize = 32,
  randomcrop_type = 'zero',
  cudnn_deterministic = false,
  optnet_optimize = true,
  generate_graph = false,
  multiply_input_factor = 1,
  widen_factor = 0.8,
  nGPU = 1,
  data_type = 'torch.CudaTensor',
  seed = 444,
  type = 'cuda'
}

local resnet = require 'wide-resnet'


do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
		local permutation = torch.randperm(input:size(1))
		for i=1,input:size(1) do		
			if 0 == permutation[i] % 3  then
				image.hflip(input[i], input[i])
			end -- need to define f
			
			if 1 == permutation[i] % 3  then
				module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float() 
				local padded = module:forward(im:float())
				local x = torch.random(1,pad*2 + 1)
				local y = torch.random(1,pad*2 + 1)

				return padded:narrow(3,x,im:size(3)):narrow(2,y,im:size(2))
			end -- need to define g
		end
      
	  --[[local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then 
			image.hflip(input[i], input[i])
			
		end
      end]]
	  
    end
	
    self.output:set(input)
    return self.output
  end
end

local function cast(t)
	print(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
model:add(cast(resnet(opt)))
model:get(2).updateGradInput = function(input) return end

--local model = resnet(opt)

--model = model:cuda()
-- criterion = nn.ClassNLLCriterion():cuda()
criterion = nn.CrossEntropyCriterion():cuda()
print('Criterion:')
print(criterion)

w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'
local batchSize = 128
local optimState = {}

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize)
		--print('i: ' .. i)
		--print(x:size())
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
						
            optim.adam(feval, w, optimState)
			
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

---------------------------------------------------------------------
function train(model, epochs, trainData, trainLabels, testData, testLabels)

	trainLoss = torch.Tensor(epochs)
	testLoss = torch.Tensor(epochs)
	trainError = torch.Tensor(epochs)
	testError = torch.Tensor(epochs)

	--reset net weights
	-- model:apply(function(l) l:reset() end)

	--timer = torch.Timer()
	best_error = 1

	for e = 1, epochs do
		trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
		print('e (b) = ' .. e)
		trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
		print('e (a) = ' .. e)
		testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
				
		print('Epoch ' .. e .. ':')
		print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
		print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
		print(confusion)
		
		torch.save('trainError.' .. prefix[1] .. '.txt', trainError)
		torch.save('testError.' .. prefix[1] .. '.txt', testError)
		
		if best_error > testError[e] then
	    	torch.save('model.' .. prefix[1] .. '.txt', model)
			best_error = testError[e]
	    end
	end

	-- plotError(trainError, testError, 'Classification Error')

end

function test(testData, testLabels)
	model = torch.load('model.txt')
	trainLoss = torch.Tensor(1)
	testLoss = torch.Tensor(1)
	trainError = torch.Tensor(1)
	testError = torch.Tensor(1)

	timer = torch.Timer()
	testLoss[1], testError[1], confusion = forwardNet(testData, testLabels, false) --test
	    
	print('Test error: ' .. testError[1], 'Test Loss: ' .. testLoss[1])
	print(confusion)

	-- plotError(trainError, testError, 'Classification Error')
end

epochs = 250

train(model, epochs, trainData, trainLabels, testData, testLabels)

--test(testData, testLabels)

--  ****************************************************************
--  Network predictions
--  ****************************************************************

--[[
model:evaluate()   --turn off dropout

print(classes[ testLabels[10] ])
print(testData[10]:size())
saveTensorAsGrid(testData[10],'testImg10.jpg')
local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 

-- assigned a probability to each classes
for i=1,predicted:size(2) do
    print(classes[i],predicted[1][i])
end

--]]

--  ****************************************************************
--  Visualizing Network Weights+Activations
--  ****************************************************************

--[[
local Weights_1st_Layer = model:get(1).weight
local scaledWeights = image.scale(image.toDisplayTensor({input=Weights_1st_Layer,padding=2}),200)
saveTensorAsGrid(scaledWeights,'Weights_1st_Layer.jpg')


print('Input Image')
saveTensorAsGrid(testData[100],'testImg100.jpg')
model:forward(testData[100]:view(1,3,32,32):cuda())
for l=1,9 do
  print('Layer ' ,l, tostring(model:get(l)))
  local layer_output = model:get(l).output[1]
  saveTensorAsGrid(layer_output,'Layer'..l..'-'..tostring(model:get(l))..'.jpg')
  if ( l == 5 or l == 9 )then
	local Weights_lst_Layer = model:get(l).weight
	local scaledWeights = image.scale(image.toDisplayTensor({input=Weights_lst_Layer[1],padding=2}),200)
	saveTensorAsGrid(scaledWeights,'Weights_'..l..'st_Layer.jpg')
  end 
end

]]