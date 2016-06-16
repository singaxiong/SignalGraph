function feat = Reader_htkfile2spectrum(files, big_endian, useGPU, precision)

for i=1:length(files)
    tmp_feat = readHTK(files{i}, [], big_endian,1);
    if strcmpi(precision, 'single')
        tmp_feat = single(tmp_feat);
    end
    dim = size(tmp_feat,1);
    dim = dim/2;

    if useGPU
        mag = gpuArray(tmp_feat(1:dim,:));
        phase = gpuArray(tmp_feat(dim+1:end,:));
    else
        mag = tmp_feat(1:dim,:);
        phase = tmp_feat(dim+1:end,:);
    end
    feat{i} = gather(exp(mag+1j*phase));
end
end