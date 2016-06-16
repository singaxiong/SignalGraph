function grad = B_logistic(input_layers, CostLayer)

[~, output, target] = prepareCostEvaluation(input_layers, CostLayer);
m = length(output);
output = sigmoid(output);

grad = -(target - output)/m;

end