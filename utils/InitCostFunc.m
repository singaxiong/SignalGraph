function cost_func = InitCostFunc(nBatch, para)

nCost = length(para.cost_func.layer_idx);

if para.useGPU
    cost_func.cost = gpuArray.zeros(1,nBatch);
    cost_func.cost_pure = gpuArray.zeros(1,nBatch);
    cost_func.subcost = gpuArray.zeros(nCost,nBatch);
    cost_func.subacc = gpuArray.zeros(nCost,nBatch);
else
    cost_func.cost = zeros(1,nBatch);
    cost_func.cost_pure = zeros(1,nBatch);
    cost_func.subcost = zeros(nCost,nBatch);
    cost_func.subacc = zeros(nCost,nBatch);
end
end