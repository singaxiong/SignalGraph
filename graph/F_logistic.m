function [cost, acc] = F_logistic(input_layers, CostLayer)

[~, output, target] = prepareCostEvaluation(input_layers, CostLayer);
m = length(output);
output = sigmoid(output);

cost = -1/m * sum( log(max(eps,output)) .* target + log(max(eps,1-output)) .* (1-target) );

acc = sum((output>0.5) == target)/m;


end