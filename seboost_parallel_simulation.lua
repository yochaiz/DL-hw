--[[ A implementation of seboost

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX.
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.optMethod`         : The base optimizaion method
- `config.momentum`          : weight for SEBOOST's momentum direction
- `config.histSize`          : number of previous directions to keep in memory
- `config.anchorPoints`      : A tensor of values, each describing the number of             iterations between an update of an anchor point
- `config.sesopUpdate`       : The number of regular optimization steps between each boosting step
- `config.sesopData`         : The training data to use for the boosting stage
- `config.sesopLabels`       : The labels to use for the boosting stage
- `config.sesopBatchSize`    : The number of samples to use for each optimization step
- `config.isCuda`            : Whether to train using cuda or cpu
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.sesopLastX`        : The last point from which the boosting was ran
- `state.itr`               : The current optimization iteration
- `state.dirs`              : The set of directions to optimize in
- `state.anchors`           : The current anchor points
- `state.aOpt`              : The current set of optimal coefficients
- `state.dirIdx`            : The next direction to override

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

require 'optim'


--[[
SV:
  We have 'config.numNodes' nodes.
  All nodes start from the same 'x'.
  Every node does 'state.nodeIters' baseMethod iterations.
  
  Once all the nodes are done, we do a sesop iteration to "merge"
  The states of the different nodes.
  
  On true parallel execution, the nodes will execute in parallel, 
  so the time for one external iteration will be:
  T_p = T(baseMethod)*state.nodeIters + T(sesop)
  
  On our simulation, the time is:
  T_s = T(baseMethod)*state.nodeIters*config.numNodes + T(sesop)
  
  Assume there are k external iteration in an epoch. 
  So assume number of internal iterations in an epoch divides state.nodeIters.
  I.e., state.nodeIters*k = number of iterations in an epoch:
  
  T_s(epoch) = (T(baseMethod)*state.nodeIters*config.numNodes + T(sesop))*k = 
    T_s(baseMethod)*(number of iterations in an epoch)*config.numNodes + T(sesop)*k
    
  T_p(epoch) = (T(baseMethod)*state.nodeIters + T(sesop))*k = 
    T(baseMethod)*(number of iterations in an epoch) + T(sesop)*k
    
  In our first expr, we assume T(sesop) is neglible, and we compute the expected parallel time as:
  
  T_p(epoch) = T_s(epoch)/config.numNodes
  
  And we plot the graphs, we should see an improvemnt as we increase the number of nodes.
]]

function optim.seboost(opfunc, x, config, state)

  -- get/update state
  local config = config or {}
  local state = state or config
  local isCuda = config.isCuda or false
  local sesopData = config.sesopData
  local sesopLabels = config.sesopLabels
  local sesopBatchSize = config.sesopBatchSize or 100
  state.sesopLastX = state.sesopLastX or x:clone() --Never forget to clone!
  state.itr = state.itr or 0
  

  local timer = torch.Timer()

  --number of iterations per node
  config.nodeIters = config.nodeIters or 100
	config.numNodes = config.numNodes or 2
  
	state.currNode = state.currNode or 0 --start from node 0
	state.lastNodeXs = state.lastNodeXs or {}
	state.splitPoint = state.splitPoint or x:clone() --the first split point is the first point
  state.sesopIteration = state.sesopIteration or 0
  
	local isMergeIter = false
  state.itr = state.itr + 1
  
  --[[
  print ('state.currNode = ' .. state.currNode)
  print ('config.nodeIters = ' .. config.nodeIters)
  print ('config.numNodes = ' .. config.numNodes)
  print ('state.itr = ' .. state.itr)
  print ('x = ')
  print(x)
  print ('state.splitPoint = ')
  print(state.splitPoint)
  ]]
	--node switch
  if (config.numNodes > 1 and state.itr % (config.nodeIters + 1) == 0) then
    print ('In node switch '.. state.itr)
		--a node has finished. Save its last x location
		state.lastNodeXs[state.currNode] = state.lastNodeXs[state.currNode] or x:clone()
    state.lastNodeXs[state.currNode]:copy(x)

		--progress to next node
		state.currNode = (state.currNode + 1)%config.numNodes

		--merge iteration (run seboost to merge).
		if (state.currNode == 0) then
			isMergeIter = true
		end

		--The new node starts from the split point
    x:copy(state.splitPoint)
  end
  

  if (isMergeIter == false or config.numNodes == 1) then
    --print ('x before opt = ')
    --print(x)
    config.optConfig[state.currNode] = config.optConfig[state.currNode] or {}
		x,fx = config.optMethod(opfunc, x, config.optConfig[state.currNode])
    --print ('x after opt = ')
    --print(x)
    --print ('--------------------------------')
		return x,fx
	end

  --Now x is the split point.
  ------------------------- SESOP Part ----------------------------
  --print ('****************SESOP***********')
  --print ('--------------------------------')
  state.dirs = torch.zeros(x:size(1), config.numNodes)
  state.aOpt = torch.zeros(config.numNodes)
  state.aOpt[1] = 1 --we start from taking the first node direction (maybe start from avrage?).
  
  if (isCuda) then
    state.dirs = state.dirs:cuda()
    state.aOpt = state.aOpt:cuda()
  end
  
  --SV, build directions matrix
  for i = 0, config.numNodes - 1 do   
    --[{ {}, i }] means: all of the first dim, slice in the second dim at i = get i col.
    state.dirs[{ {}, i + 1 }]:copy(state.lastNodeXs[i] - state.splitPoint) 
  end


  --now optimize!
  local xInit = state.splitPoint
    -- create mini batch
  local subT = (state.sesopIteration) * sesopBatchSize + 1
  subT = subT % (sesopData:size(1) - sesopBatchSize) --Calculate the next batch index
  local sesopInputs = sesopData:narrow(1, subT, sesopBatchSize)
  local sesopTargets = sesopLabels:narrow(1, subT, sesopBatchSize)
  if isCuda then
    sesopInputs = sesopInputs:cuda()
    sesopTargets = sesopTargets:cuda()
  end

  -- Create inner opfunc for finding a*
  local feval = function(a)
    --A function of the coefficients
    local dirMat = state.dirs
    --Note that opfunc also gets the batch
    local afx, adfdx = opfunc(xInit + dirMat*a, sesopInputs, sesopTargets)
    return afx, (dirMat:t()*adfdx)
  end

  --x,f(x)
  local _, fHist = optim.cg(feval, state.aOpt, config, state) --Apply optimization using inner function
   
  --updating model weights!
  x:copy(xInit)
  local sesopDir = state.dirs*state.aOpt 
  x:add(sesopDir)
  
  --the new split point is 'x'.
  --The next time this function is called will be with 'x'.
  --The next time we will change a node, it will get this 'x'.
  state.splitPoint:copy(x)
  
  state.sesopIteration = state.sesopIteration + 1
  return x,fHist
  
end

return optim

