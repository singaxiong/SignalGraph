function [grad] = B_cross_entropy(input_layers)
[m, output, target] = prepareCostEvaluation(input_layers);
% cost = -1/m * sum( sum( log(output) .* target ) );
grad = -1/m * target ./ output;
end