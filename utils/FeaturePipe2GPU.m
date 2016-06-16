function processing = FeaturePipe2GPU(processing)


for i=1:length(processing)
    if isfield(processing{i}, 'transform')
        processing{i}.transform = gpuArray(processing{i}.transform);
    end
    if isfield(processing{i}, 'bias')
        processing{i}.bias = gpuArray(processing{i}.bias);
    end
end
