function mb = GetMinibatch(data, mb_idx, useGPU)
mb = data(:, mb_idx);
if useGPU
    for si=1:length(mb)
        if strcmpi(class(mb{si}), 'gpuArray')==0
            mb{si} = gpuArray(mb{si});
        end
    end
end
end