function [cost, acc] = F_multi_logistic(input_layers, CostLayer)

[nSeg, output, target, scale, nFrOrig, mask] = prepareCostEvaluation(input_layers, CostLayer);
m = size(output,2);
output = sigmoid(output);
output = output(:);
target = target(:);

cost = -1/m * sum( log(max(eps,output)) .* target + log(max(eps,1-output)) .* (1-target) );

acc = sum((output>0.5) == target)/length(target);


end