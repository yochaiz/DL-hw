require 'rnn'
require 'cunn'
require 'nngraph'
local dl = require 'dataload'

opt = lapp[[
  -m,--model                (default "")      model to load
  -s,--sentence             (default "buy low sell high is the")      sentence
  -n                        (default 20)      num of words
  --forget_every_word       (default 0)      if 1 call forget after each predict.
]]

batchSize = 32
local trainset, validset, testset = dl.loadPTB({batchSize,1,1})
vocabulary = trainset.ivocab
--local model = torch.load('/home/yochaiz@st.technion.ac.il/hw3/save/rnnlm/ptb:ml10:1484574476:1.t7')
local model = torch.load('save/rnnlm/'..opt.model.. '.t7')
print(model.model)

function word2index(word) 
	for k, v in pairs(vocabulary) do
		if (v == word) then
			return k
		end
	end
	return nil
end

function sentence2indexes(sentence)
	local words = sentence:split(' ')
	-- print(words)
	for i=1,#words do
		words[i] = word2index(words[i])
	end
	-- print(words)
	-- return torch.Tensor(words)
	return words
end

model.model:evaluate()

--i means the ith choise
function predict(sentence, i)
  if opt.forget_every_word == 1 then
    model.model:forget()
  end
  
  output = model.model:forward(torch.Tensor(sentence2indexes(sentence)))
  sorted, indices = torch.sort(output[#output])
  
  local res = nil
  while res == '<unk>' or res == 'N' or res == '$' or res == nil do
    res = vocabulary[(indices[1][indices:size(2) - i])]
    i = i + 1
  end
  --print('sentence='..sentence)
  --print ('chosen i = '..(i - 1))
  return res
end

function generateSentece(idx)
  model.model:forget()
  sentence = opt.sentence
  for i = 1,opt.n do
    if i == 1 then
      word = predict(sentence, idx)
    else
      word = predict(sentence, 0)
    end

    sentence = sentence..' '..word
    if (word == '<eos>') then
      break
    end
  end
  print(sentence)
end

for i=1,10 do
  generateSentece(i - 1)
end

print(model.opt.seqlen)

--print(output)
--val,idx = torch.max(output[1])
--print('val:[' .. val .. '], idx:[' .. idx .. ']')
