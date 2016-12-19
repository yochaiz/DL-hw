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
require 'seboost_parallel_simulation'
--require 'avrage_parallel_simulation'

function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

--[[local trainset = torch.load('/home/yochaiz/CIFAR-10/Datasets/cifar10-train.t7')
local testset = torch.load('/home/yochaiz/CIFAR-10/Datasets/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)
]]

local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
local testData = mnist.testdataset().data:float();
local testLabels = mnist.testdataset().label:add(1);

local mean = trainData:mean()
local std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);

trainData = trainData:reshape(trainData:size(1), 1, trainData:size(2), trainData:size(3))
--trainLabels = trainLabels:reshape(trainLabels:size(1), 1, trainLabels:size(2), trainLabels:size(3))
testData = testData:reshape(testData:size(1), 1, testData:size(2), testData:size(3))
--trainData = trainData:reshape(trainData:size(1), 1, trainData:size(2), trainData:size(3))



print(trainData:size())
print(trainLabels:size())

saveTensorAsGrid(trainData:narrow(1,100,36),'train_100-136.jpg') -- display the 100-136 images in dataset
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
-]]

--  ****************************************************************
--  Full Example - Training a ConvNet on Cifar10
--  ****************************************************************

-- Load and normalize data:
--[[
local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
print(#redChannel)

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

]]

--  ****************************************************************
--  Define our neural network
--  ****************************************************************

local outputSize = 10

fullySize = 90

local model = nn.Sequential()

model:add(cudnn.SpatialConvolution(1, 32, 5, 5)) -- 1 input image channel, 32 output channels, 5x5 convolution kernel
model:add(cudnn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
model:add(cudnn.ReLU(true))                          -- ReLU activation function
model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(cudnn.SpatialConvolution(32, 64, 3, 3))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
model:add(cudnn.ReLU(true))
model:add(nn.SpatialBatchNormalization(64))
model:add(cudnn.SpatialConvolution(64, 32, 3, 3))
model:add(nn.View(32*3*3):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
model:add(nn.Linear(32*3*3, fullySize))             -- fully connected layer (matrix multiplication between input and weights)
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
model:add(nn.Linear(fullySize, outputSize))            -- 10 is the number of outputs of the network (in this case, 10 digits)
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classificati

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()



--64 channels, 32 kernels, 3x3. 

w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    --print(RandOrder)
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

function getCurriculum(data, labels)
    local lossAcc = 0
    local numBatches = 0
    model:evaluate() 
    loss = torch.Tensor(data:size(1))
    local batchSize = 128
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()		
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)

	--print(y:size())
	for j=1,y:size(1) do
        	local err = criterion:forward(y[j], yt[j])
		--print ('batchSize = ' .. batchSize .. ', j = ' .. j .. ', numBatches = ' .. numBatches)
		--print((numBatches - 1)*batchSize + j)
		loss[(numBatches - 1)*batchSize + j] = err
	end


    end
    --print (loss)
    
    y,i = torch.sort(loss)

    return data:index(1,i), labels:index(1,i)
end

function loadCurriculum(data, labels)
	i = torch.load('curriculum.txt')

	--1000 random swaps	
	local k=1000
	for j=1,k do
		local i1 = (torch.random() % data:size(1)) + 1
		local i2 = (torch.random() % data:size(1)) + 1
		local temp = i[i1]
		i[i1] = i[i2]
		i[i2] = temp
	end
	return data:index(1,i), labels:index(1,i)
end

function dumpCurriculum(data, labels)
    local lossAcc = 0
    local numBatches = 0
    model:evaluate() 
    loss = torch.Tensor(data:size(1))
    local batchSize = 128
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()		
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)

	--print(y:size())
	for j=1,y:size(1) do
        	local err = criterion:forward(y[j], yt[j])
		--print ('batchSize = ' .. batchSize .. ', j = ' .. j .. ', numBatches = ' .. numBatches)
		--print((numBatches - 1)*batchSize + j)
		loss[(numBatches - 1)*batchSize + j] = err
	end


    end
    --print (loss)
    
    y,i = torch.sort(loss)
    torch.save('curriculum.txt', i)

    return data:index(1,i), labels:index(1,i)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'
--learningRate=0.05
local batchSize = 128

function forwardNet(data, labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
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
        local x = data:narrow(1, i, batchSize):cuda()		
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        --w, dE_dw = model:getParameters()
        if train then
            --w, inputs, targets
            function feval(_w, inputs, targets)
              
                if w ~= _w then 
                  --because we use (*) to calc f(x), we need to make sure the model has the current parameters
                  --to calc f(_w) and not f(w)
                  w:copy(_w) --Needs to be copied before eval
                end
            
                local x = inputs or x --take inputs in case of sesop and x incase of baseMethod.
                local yt = targets or yt
                local y = model:forward(x) --(*)
                local err = criterion:forward(y, yt)
                
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
      --giving w to adadelta allows it to change it inplace.
	    --optim.adadelta(feval, w, optimState)
      
	    optim.seboost(feval, w, sesopConfig, optimState)
            --optim.adagrad(feval, w, optimState)
	    --optim.sgd(feval, w, optimState)
        end
    end

    if train then
    	lossAcc = 0
    
    	for i = 1, data:size(1) - batchSize, batchSize do
        	local x = data:narrow(1, i, batchSize):cuda()		
        	local yt = labels:narrow(1, i, batchSize):cuda()
        	local y = model:forward(x)
        	local err = criterion:forward(y, yt)
        	lossAcc = lossAcc + err
    	end
    end

    confusion:updateValids()
    --local avgLoss = lossAcc / numBatches
    local avgLoss = lossAcc / data:size(1)
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

---------------------------------------------------------------------

function train(epochs, trainData, trainLabels, testData, testLabels)
  
  trainLoss = {}
  testLoss = {}
  trainError = {}
  testError = {}
  epochTimes = {}
  
  for numOfNodes = 1, 5 do
    optimState = {}
    sesopConfig = {
      --optMethod=optim.adadelta, 
      optMethod=optim.adagrad, 
      histSize=10,
      sesopData=trainData,
      sesopLabels=trainLabels,
      isCuda=true,
      optConfig={}, --state for the inner optimization function.
      sesopUpdate=400, --sesop iteration every 400 'optMethod' iterations.
      sesopBatchSize=1000,
      numNodes=math.pow(2, numOfNodes - 1),
      nodeIters=math.ceil(100/numOfNodes),
    }

    epochTimes[sesopConfig.numNodes] = torch.Tensor(epochs + 1)
    trainLoss[sesopConfig.numNodes] = torch.Tensor(epochs + 1)
    testLoss[sesopConfig.numNodes] = torch.Tensor(epochs + 1)
    trainError[sesopConfig.numNodes] = torch.Tensor(epochs + 1)
    testError[sesopConfig.numNodes] = torch.Tensor(epochs + 1)

    torch.manualSeed(8765467)
    model:apply(function(l) l:reset() end)
    
    testLoss[sesopConfig.numNodes][1], testError[sesopConfig.numNodes][1], confusion = forwardNet(testData, testLabels, false) 
    
    best_error = 1

    for e = 1, epochs*sesopConfig.numNodes do
        e = math.ceil(e/sesopConfig.numNodes)
        trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
        --trainData, trainLabels = loadCurriculum(trainData, trainLabels)
        print('e = ' .. e)
        timer = torch.Timer()
        trainLoss[sesopConfig.numNodes][e + 1], trainError[sesopConfig.numNodes][e + 1] = forwardNet(trainData, trainLabels, true) --train
        timer:stop()
        epochTimes[sesopConfig.numNodes][e + 1] = timer:time().real
  
        testLoss[sesopConfig.numNodes][e + 1], testError[sesopConfig.numNodes][e + 1], confusion = forwardNet(testData, testLabels, false) --test
       
        if e % 5 == 0 then
          print('Epoch ' .. e .. ':')
          print('Training error: ' .. trainError[sesopConfig.numNodes][e + 1], 'Training Loss: ' .. trainLoss[sesopConfig.numNodes][e + 1])
          print('Test error: ' .. testError[sesopConfig.numNodes][e + 1], 'Test Loss: ' .. testLoss[sesopConfig.numNodes][e + 1])
          print(confusion)
        end

        if best_error > testError[sesopConfig.numNodes][e + 1] then
          torch.save('model.txt', model)
          best_error = testError[sesopConfig.numNodes][e + 1]
          --dumpCurriculum(trainData, trainLabels)
        end
        torch.save('testError.txt', testError)
        torch.save('trainError.txt', trainError)
        torch.save('epochTimes.txt', epochTimes)
          
    end

    --print ('Best error ' .. (1 - best_error))
    --plotLoss(trainLoss[sesopConfig.numNodes], testLoss[sesopConfig.numNodes], 'Loss')
    
  end
  --plotError(trainError, testError, 'Classification Error')
  --plotTimeToAccuracy(testError, 'Time to acc')
end


function plotLoss(trainLoss, testLoss, title)
	require 'gnuplot'
	local range = torch.range(1, trainLoss:size(1))
	gnuplot.pngfigure('testVsTrainLoss.png')
	gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Loss')
	gnuplot.plotflush()
end

function plotTimeToAccuracy(testError, title)
  require 'gnuplot'
  gnuplot.pngfigure('timeToAccuracy.png')
  gnuplot.xlabel('Number of nodes')
  gnuplot.ylabel('Time (epochs)')
  
  local testErrorSize = 0
  for numOfNodes, error in pairs(testError) do 
    testErrorSize = testErrorSize + 1
  end
  
  timeToAccuracy = torch.Tensor(testErrorSize)
  epochs = torch.Tensor(testErrorSize)
  
  for numOfNodes = 1, testErrorSize do
    epochs[numOfNodes] = math.pow(2, numOfNodes - 1)
    for time = 1, testError[numOfNodes]:size(1) do
      
      if (testError[numOfNodes][time] < 0.007) then
        timeToAccuracy[numOfNodes] = time
        break
      end
      
    end
    
  end
  
  gnuplot.plot({'Time to accuracy 0.007', timeToAccuracy})
  gnuplot.plotflush()
end

function plotError(trainError, testError, title)
	require 'gnuplot'
  gnuplot.pngfigure('testVsTrainError.png')
  
  --for numOfNodes, error in pairs(testError) do 
    --local range = torch.range(1, error:size(1))  
    --gnuplot.plot({'testError(' .. numOfNodes .. ')', error})
  --end
  
  gnuplot.plot({'testError(1)', testError[1]}, 
    {'testError(2)', testError[2]},
    {'testError(4)', testError[3]},
    {'testError(8)', testError[4]})
  
  --gnuplot.plot({'testError(1)', testError[1]}, 
  --  {'testError(2)', testError[2]})
  
  gnuplot.xlabel('epochs')
  gnuplot.ylabel('Error')
  gnuplot.plotflush()
    --[[
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
  ]]
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

	plotError(trainError, testError, 'Classification Error')
end


train(7, trainData, trainLabels, testData, testLabels)
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
