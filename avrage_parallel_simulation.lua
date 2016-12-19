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
  We have 'state.numNodes' nodes.
  All nodes start from the same 'x'.
  Every node does 'state.nodeIters' baseMethod iterations.
  
  Once all the nodes are done, we do a "avrage" iteration to "merge"
  The states of the different nodes.
  
  On true parallel execution, the nodes will execute in parallel, 
  so the time for one external iteration will be:
  T_p = T(baseMethod)*state.nodeIters + T(sesop)
  
  On our simulation, the time is:
  T_s = T(baseMethod)*state.nodeIters*state.numNodes + T(sesop)
  
  Assume there are k external iteration in an epoch. 
  So assume number of internal iterations in an epoch divides state.nodeIters.
  I.e., state.nodeIters*k = number of iterations in an epoch:
  
  T_s(epoch) = (T(baseMethod)*state.nodeIters*state.numNodes + T(sesop))*k = 
    T_s(baseMethod)*(number of iterations in an epoch)*state.numNodes + T(sesop)*k
    
  T_p(epoch) = (T(baseMethod)*state.nodeIters + T(sesop))*k = 
    T(baseMethod)*(number of iterations in an epoch) + T(sesop)*k
    
  In our first expr, we assume T(sesop) is neglible, and we compute the expected parallel time as:
  
  T_p(epoch) = T_s(epoch)/state.numNodes
  
  And we plot the graphs, we should see an improvemnt as we increase the number of nodes.
]]

function optim.seboost(opfunc, x, config, state)

  -- get/update state
  local config = config or {}
  local state = state or config
  local momentum = config.momentum or 0.9
  local histSize = config.histSize
  local anchorPoints = config.anchorPoints or nil
  local isCuda = config.isCuda or false
  local sesopUpdate = config.sesopUpdate or 100
  local sesopData = config.sesopData
  local sesopLabels = config.sesopLabels
  local sesopBatchSize = config.sesopBatchSize or 100
  local eps = 1e-5 --Minimal norm of a direction
  state.sesopLastX = state.sesopLastX or x:clone() --Never forget to clone!
  state.itr = state.itr or 0
  state.itr = state.itr + 1

  local timer = torch.Timer()

  --number of iterations per node
  state.nodeIters = state.nodeIters or 100
	state.numNodes = state.numNodes or 2
	state.currNode = state.currNode or 0 --start from node 0
	state.lastNodeXs = state.lastNodeXs or {}
	state.splitPoint = state.splitPoint or x --the first split point is the first point

	local isMergeIter = false

	--node switch
  if (state.numNodes > 1 and state.itr % state.nodeIters == 0) then
		--a node has finished. Save its last x location
		state.lastNodeXs[state.currNode] = x

		--progress to next node
		state.currNode = (state.currNode + 1)%state.numNodes

		--merge iteration (run seboost to merge).
		if (state.currNode == 0) then
			isMergeIter = true
		end

		--The new node starts from the split point
		x = state.splitPoint
  end

  if (isMergeIter == false or state.numNodes == 1) then
		x,fx = config.optMethod(opfunc, x, config.optConfig)
		return x,fx
	end

  ------------------------- Avrage Part ----------------------------

  
  --SV, calc the avrage
  local newX = state.lastNodeXs[0];
  for i = 1, state.numNodes - 1 do   
    newX = newX + state.lastNodeXs[i]
  end
  newX = newX/state.numNodes

  x:copy(newX)
  
  --the new split point is 'x'.
  --The next time this function is called will be with 'x'.
  --The next time we will change a node, it will get this 'x'.
  state.splitPoint = x
  
  x,fx = config.optMethod(opfunc, x, config.optConfig)
  return x,fx
  
end

return optim

