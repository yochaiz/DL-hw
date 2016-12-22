-- This is a modified version of NIN network in
-- https://github.com/szagoruyko/cifar.torch
-- Network-In-Network: http://arxiv.org/abs/1312.4400
-- Modifications:
--  * removed dropout
--  * added BatchNorm
--  * the last layer changed from avg-pooling to linear (works better)
require 'nn'
local utils = paths.dofile'utils.lua'

local function createModel(opt)
   local model = nn.Sequential()

   local function Block(...)
     local arg = {...}
     model:add(nn.SpatialConvolution(...):noBias())
     model:add(nn.SpatialBatchNormalization(arg[2],1e-5))
     model:add(nn.ReLU(true))
     return model
   end

   local d1 = 64
   local d2 = 32
   local d3 = 45
   local d4 = 28
   local d5 = 24
   
   Block(3,d1,5,5,1,1,2,2)
   Block(d1,d2,1,1)
   Block(d2,d3,1,1)
   model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
   Block(d3,d4,5,5,1,1,2,2)
   Block(d4,d4,1,1)
   Block(d4,d4,1,1)
   model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
   Block(d4,d5,3,3,1,1,1,1)
   Block(d5,d5,1,1)
   Block(d5,d5,1,1)
   model:add(nn.SpatialAveragePooling(8,8,1,1))
   model:add(nn.View(-1):setNumInputDims(3))
   --model:add(nn.Linear(192,opt and opt.num_classes or 10))
   model:add(nn.Linear(d5, 10))
   utils.FCinit(model)
   utils.testModel(model)
   utils.MSRinit(model)
   return model
end

return createModel
