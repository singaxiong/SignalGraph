function block_size = decide_max_block_size_tree2(layer, para)

if para.useGPU == 1 && para.storeDataInCell == 0    % if we store data in cell, we move data to GPU in minibatch, so we are usually not limited by the GPU memory when deciding block size
    gpu = gpuDevice;
    GPU_max_memory =  gpu.FreeMemory; % This is the maximum number of double precision numbers that can be stored in the GPU
    max_memory = GPU_max_memory/2;
else
    max_memory = 2e10;  % assume we can use 10GB
end

% Compute how much training data can we put into GPU
inputDim = 0; 
for i=1:length(layer)
    if strcmpi(layer{i}.name, 'input')
        if strcmpi(layer{i+layer{i}.next(1)}.name, 'cross_entropy'); continue; end    % don't count class labels as they do not occupied much memory.
        if para.IO.sparse(layer{i}.inputIdx)
            inputDim = inputDim + layer{i}.dim(1)/10;
        else
            inputDim = inputDim + layer{i}.dim(1);
        end
    end
end

max_sample = max_memory / inputDim;

if para.singlePrecision==0
    block_size = max_sample / 8 / 4;   % use only half of the memory
else
    block_size = max_sample / 4 / 4;
end

if isfield(para.IO, 'blockSizeMultiplier')
    block_size = block_size * para.IO.blockSizeMultiplier;
end
end