function grad = B_multi_logistic(input_layers, CostLayer)

[nSeg, output, target, scale, nFrOrig, mask] = prepareCostEvaluation(input_layers, CostLayer);
m = size(output,2);
output = sigmoid(output);

grad = -(target - output)/m;

grad = PostprocessCostEvaluation(grad, output, mask, nSeg, nFrOrig, CostLayer);

end