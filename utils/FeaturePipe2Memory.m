function processing = FeaturePipe2Memory(processing)

for i=1:length(processing)
    if isfield(processing{i}, 'name') && strcmpi(processing{i}.name, 'affinetransform')
        processing{i}.transform = gather(processing{i}.transform);
        if isfield(processing{i}, 'bias')
            processing{i}.bias = gather(processing{i}.bias);
        end
    end
end
