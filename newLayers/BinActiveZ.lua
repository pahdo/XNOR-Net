local BinActiveZ , parent= torch.class('nn.BinActiveZ', 'nn.Module')
-- https://github.com/torch/nn/blob/master/doc/module.md

-- Output = Sign(Input)
function BinActiveZ:updateOutput(input)
	local s = input:size()
   self.output:resizeAs(input):copy(input)
   self.output=self.output:sign();
   return self.output
end

-- Grad output is same as regular based on real-valued weights for [-1, 1]
-- Grad output is 0 for (-infty, -1) U (1, infty)
function BinActiveZ:updateGradInput(input, gradOutput)
   local s = input:size()
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput[input:ge(1)]=0
   self.gradInput[input:le(-1)]=0
   return self.gradInput
end