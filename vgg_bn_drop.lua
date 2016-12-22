require 'nn'

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

local depth1 = 12
local depth2 = 16
local depth3 = 32
local depth4 = math.floor(0.85 * depth3)
local depth5 = depth4
local depth6 = depth5

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

local drop1 = 0.3
local drop2 = 0.4
local drop3 = 0.5

ConvBNReLU(3,depth1):add(nn.Dropout(drop1))
ConvBNReLU(depth1,depth1)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(depth1,depth2):add(nn.Dropout(drop2))
ConvBNReLU(depth2,depth2)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(depth2,depth3):add(nn.Dropout(drop2))
-- ConvBNReLU(depth3,depth3):add(nn.Dropout(drop2))
ConvBNReLU(depth3,depth3)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(depth3,depth4):add(nn.Dropout(drop2))
-- ConvBNReLU(depth4,depth4):add(nn.Dropout(drop2))
ConvBNReLU(depth4,depth4)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(depth4,depth5):add(nn.Dropout(drop2))
-- ConvBNReLU(depth5,depth5):add(nn.Dropout(drop2))
ConvBNReLU(depth5,depth5)
vgg:add(MaxPooling(2,2,2,2):ceil())
vgg:add(nn.View(depth5))

classifier = nn.Sequential()
classifier:add(nn.Dropout(drop3))
classifier:add(nn.Linear(depth5,depth6))
classifier:add(nn.BatchNormalization(depth6))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(drop3))
classifier:add(nn.Linear(depth6,10))
vgg:add(classifier)
-- vgg:add(nn.LogSoftMax())  

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init('nn.SpatialConvolution')
end

local function FCinit(net,name)
	for k,v in pairs(net:findModules(name))
		v.bias:zero()
		v.weight:zero()
	end
end

MSRinit(vgg)
-- FCinit(vgg,'nn.Linear')

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg