local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

--We'll start by normalizing our data
local mean = trainData:mean()
local std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);

----- ### Shuffling data

function shuffle(data, labels) --shuffle data function
    local randomIndexes = torch.randperm(data:size(1)):long() 
    return data:index(1,randomIndexes), labels:index(1,randomIndexes)
end

------   ### Define model and criterion

require 'nn'
require 'cunn'
require 'cudnn'

--local inputSize = 28*28
local outputSize = 10
-- local layerSize = {inputSize, 64,128,256}
--local layerSize = {inputSize, 75, 60, 50}
--local layerSize = {inputSize, 67, 67, 100}
-- local layerSize = {inputSize, 70, 70, 100} -- 98.00 %

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
model:add(nn.View(32*4*4):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
model:add(nn.Linear(32*4*4, 256))             -- fully connected layer (matrix multiplication between input and weights)
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
model:add(nn.Linear(256, outputSize))            -- 10 is the number of outputs of the network (in this case, 10 digits)
model:add(nn.LogSoftMax())

--[[model = nn.Sequential()
model:add(nn.View(28 * 28)) --reshapes the image into a vector without copy
-- model:add(nn.Dropout(0.2):cuda(), 8)
for i=1, #layerSize-1 do
    model:add(nn.Linear(layerSize[i], layerSize[i+1]))	
	model:add(nn.ReLU())
	
	local tt = torch.randperm(3);
	if tt:select(1,1) == 1 then
		model:add(nn.ReLU())
	elseif tt:select(1,1) == 2 then
		-- model:add(nn.Tanh())
		model:add(nn.Sigmoid())
	else
		model:add(nn.Sigmoid())
	end
end

model:add(nn.Linear(layerSize[1], layerSize[2]))	
model:add(nn.ReLU())
model:add(nn.Linear(layerSize[2], layerSize[3]))	
model:add(nn.ReLU())
model:add(nn.Linear(layerSize[3], layerSize[4]))	
model:add(nn.ReLU())

model:add(nn.Linear(layerSize[#layerSize], outputSize))
model:add(nn.LogSoftMax())   -- f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
-]]

model:cuda() --ship to gpu
print(tostring(model))

local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement()) --over-specified model


---- ### Classification criterion
criterion = nn.CrossEntropyCriterion():cuda()
-- criterion = nn.ClassNLLCriterion():cuda()

---	 ### predefined constants

require 'optim'

--- ### Main evaluation + training function

function forwardNet(data, labels, train)
	timer = torch.Timer()

    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        -- lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            -- optim.sgd(feval, w, optimState)
			optim.adagrad(feval, w, optimState)
        end
    end
    
	numBatches = 0
	for i = 1, data:size(1) - batchSize, batchSize do
		numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
	end
	
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
	-- print(timer:time().real .. ' seconds')

    return avgLoss, avgError, confusion
end


--- ### Train the network on training set, evaluate on separate set

require 'gnuplot'

epochs = 100
batchSize = 128
local range = torch.range(1, epochs)

--[[optimState = {
    learningRate = 0.35
}]]

--[[lr = {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5}
moment = {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9}
wd = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 5e-1, 5e-2, 5e-3, 5e-4, 5e-5}
]]

lr = {0.25}
moment = {0.9}
wd = {0.0001}

results = {}
maxCorrectTotal = 0
max_i = 0
max_j = 0
max_k = 0

for i = 1, table.getn(lr) do
	results[i] = {}
	for j = 1, table.getn(moment) do
		results[i][j] = {}
		for k = 1, table.getn(wd) do		
			trainLoss = torch.Tensor(epochs)
			testLoss = torch.Tensor(epochs)
			trainError = torch.Tensor(epochs)
			testError = torch.Tensor(epochs)
			maxCorrectLocal = 0
		
			optimState = {
				learningRate = lr[i],
				momentum = moment[j],
				weightDecay = wd[k]    
			}
			
			--[[optimState = {
				learningRate = 0.3,
				momentum = 0.3,
				weightDecay = 1e-5    
			}]]

			--reset net weights
			model:apply(function(l) l:reset() end)

			for e = 1, epochs do
				trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
				trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
				testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
				
				if (confusion.totalValid > maxCorrectLocal) then
					maxCorrectLocal = confusion.totalValid;
				end
				
				if e % 5 == 0 then
					print('Epoch ' .. e .. ':')
					print('Training err: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
					print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
					print(tostring(confusion))
					print('---')
				end
			end
			
			results[i][j][k] = maxCorrectLocal;
			
			if (maxCorrectLocal > maxCorrectTotal) then
				maxCorrectTotal = maxCorrectLocal
				max_i = i
				max_j = j
				max_k = k
				
				gnuplot.pngfigure('test_' .. lr[i] .. '_' .. moment[j] .. '_' .. wd[k] .. '.png')
				gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss},{'testError',testError},{'trainError',trainError})
				gnuplot.xlabel('epochs')
				gnuplot.ylabel('Loss/Error')
				gnuplot.plotflush()
			end				

			print('lr:[' .. lr[i] .. '] , momentum:[' .. moment[j] .. '] , wDecay:[' .. wd[k] .. ']')
			print('Max correct: [' .. maxCorrectLocal*100 .. ']')
			print('Total max so far: [' .. maxCorrectTotal*100 .. ']')
			print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')			
		end
	end
end

for i = 1, table.getn(lr) do	
	for j = 1, table.getn(moment) do		
		for k = 1, table.getn(wd) do
			print('i:[' .. i .. '],j:' .. j .. '],k:[' .. k .. '] = [' .. results[i][j][k] .. ']')
		end
	end
end


---		### Introduce momentum, L2 regularization
--reset net weights
--[[
model:apply(function(l) l:reset() end)

optimState = {
    learningRate = 0.1,
    momentum = 0.9,
    weightDecay = 1e-3
    
}
for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
end

print('Training error: ' .. trainError[epochs], 'Training Loss: ' .. trainLoss[epochs])
print('Test error: ' .. testError[epochs], 'Test Loss: ' .. testLoss[epochs])
]]




--- ### Insert a Dropout layer
--[[
model:insert(nn.Dropout(0.9):cuda(), 8)
]]




-- ********************* Plots *********************
--[[
require 'gnuplot'
local range = torch.range(1, epochs)
gnuplot.pngfigure('test.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()
]]














