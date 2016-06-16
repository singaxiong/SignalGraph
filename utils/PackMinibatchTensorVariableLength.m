function data = PackMinibatchTensorVariableLength(feat, nSeqInMB, nSeq, isVAD)
nBatch = ceil(nSeq/nSeqInMB);
nStream = size(feat,1);

if size(feat,2)<nSeq    % some cell have more than 1 sequence. we need to make each cell contains just 1 seq first. 
    feat2 = cell(size(feat,1), nSeq);
    % to be implemented later
    
end

randSegIdx = randperm(nSeq);
feat(:,randSegIdx) = feat;
for i=1:nStream
    if isVAD(i); continue; end
    for j=1:nBatch
        curr_feat = feat(i, nSeqInMB*(j-1)+1 : min(nSeq,nSeqInMB*j));
        data{i,j} = cell2mat_tensor3D(curr_feat, -1e10);
    end
end

end
