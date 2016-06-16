function mb = GetMinibatch2(minibatch, para, mb_idx)

% check whether data are in GPU, if not, move them to GPU

if isfield(minibatch, 'idx1')   % if the minibatches are put in a big matrix and indexed by start and top indexes
    for si=1:length(minibatch.data)
        mb{si} = minibatch.data{si}(:,minibatch.idx1(mb_idx,si):minibatch.idx2(mb_idx,si));
    end
else        % if the minibathes are in cell arrays
    mb = minibatch.data(:, mb_idx);
end

if para.useGPU
    for si=1:length(mb)
        if strcmpi(class(mb{si}), 'gpuArray')==0
            mb{si} = gpuArray(mb{si});
        end
    end
end

end