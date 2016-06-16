function block_size = decide_max_block_size_RBM(para)

if para.useGPU == 1
    gpu = gpuDevice;
    GPU_max_memory =  gpu.FreeMemory/8; % This is the maximum number of double precision numbers that can be stored in the GPU
    % Compute how much training data can we put into GPU
    inputDim = para.layerSize(1);
    outputDim = para.layerSize(end);
    max_sample = GPU_max_memory / (inputDim + outputDim);
    block_size = max_sample / 5;   % Let's be conservative, don't use all the memory of GPU
else
    block_size = 1e5;
end
