
require 'image'
require 'nn'

function saveTensorAsGrid(tensor,fileName) 
	local padding = 0
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local testset = torch.load('/home/yochaiz@st.technion.ac.il/cifar.torch/cifar10-test.t7')
local testData = testset.data:float()

local immg = testData:narrow(1,100,1):select(1,1)
saveTensorAsGrid(immg,'train_100.jpg') -- display the 100-136 images in dataset

local immgFlipH = image.hflip(immg)
saveTensorAsGrid(immgFlipH,'train_100_Hflip.jpg')

local immgFlipV = image.vflip(immg)
saveTensorAsGrid(immgFlipV,'train_100_Vflip.jpg')

local pad = 25
module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float() 
-- module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
local padded = module:forward(immg:float())
local x = torch.random(1,pad*2 + 1)
local y = torch.random(1,pad*2 + 1)
-- image.save('train_100_ZeroPadded.jpg', padded)
saveTensorAsGrid(padded,'train_100_padded.jpg')

reflection = padded:narrow(3,x,immg:size(3)):narrow(2,y,immg:size(2))
saveTensorAsGrid(reflection,'train_100_reflection.jpg')



