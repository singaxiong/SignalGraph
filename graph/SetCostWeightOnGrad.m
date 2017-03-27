function weightedGrad = SetCostWeightOnGrad(grad, cost_func, layer_idx)

% check whether current layer is a cost layer, if yes, apply the cost
% weight
isCostLayer = cost_func.layer_idx == layer_idx;
if sum(isCostLayer)
    costWeight = cost_func.layer_weight(isCostLayer);
    if iscell(grad)
        for i=1:length(grad)
            weightedGrad{i} = grad{i} * costWeight;
        end
    else
        weightedGrad = grad * costWeight;
    end
end
end

