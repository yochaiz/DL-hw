# My project's README

local layerSize = {inputSize, 67, 67, 100}

model = nn.Sequential()

model:add(nn.View(28 * 28)) --reshapes the image into a vector without copy

for i=1, #layerSize-1 do

    model:add(nn.Linear(layerSize[i], layerSize[i+1]))	
	
	model:add(nn.ReLU())
	
end

criterion = nn.ClassNLLCriterion():cuda()

optim.sgd(feval, w, optimState)

epochs = 200
batchSize = 128

optimState = {
	learningRate = 0.35,
	momentum = 0.7,
	weightDecay = 5e-05    
}

lr:[0.35] , momentum:[0.7] , wDecay:[5e-05]	

