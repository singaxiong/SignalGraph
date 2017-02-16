function [output, mask] = F_SpatialCov(input_layer, curr_layer)

input = input_layer.a;
[D,T,N] = size(input);

curr_layer = SetDefaultValue(curr_layer, 'winSize', 0);
curr_layer = SetDefaultValue(curr_layer, 'winShift', 1);

nBin = length(curr_layer.freqBin);
nCh = D/nBin;

input2 = reshape(input, nBin, nCh, T, N);

if curr_layer.winSize == 0
    nf = 1;
else
    nf = fix((T-curr_layer.winSize+curr_layer.winShift)/curr_layer.winShift);
end
mask = zeros(nf, N, 'like', real(input2));

if N==1
%     R = ComplexSpectrum2SpatialCov(input2, curr_layer.winSize, curr_layer.winShift);
% %     output = permute(R, [3 1 2 4]);
% %     output = reshape(output, nBin*nCh^2, size(output,4),N);
%     output = reshape(R, nCh^2*nBin, size(R,4),N);
    
    X2 = permute(input2, [2 1 3]);
    XX = outProdND(X2);
    XX2 = reshape(XX, nCh^2*nBin, T);
    
    if curr_layer.winSize == 0
        output = squeeze(mean(XX2, 2));
    else
%         idx = [ones(1,half_ctx) 1:T ones(1,half_ctx)*T];
        SCM = conv2(XX2, ones(1,curr_layer.winSize, class(gather(input2)))/curr_layer.winSize, 'valid');
        output = SCM(:, 1:curr_layer.winShift:end);
    end
    
else
    X2 = permute(input2, [2 1 3 4]);
    XX = outProdND(X2);
    XX2 = reshape(XX, nCh^2*nBin, T, N);
    
    if curr_layer.winSize == 0
        output = mean(XX2, 2);
    else
% %         idx = [ones(1,half_ctx) 1:T ones(1,half_ctx)*T];
%         XX3 = reshape(permute(XX2, [1 3 2]), nCh^2*nBin*N, T);
%         SCM = conv2(XX3, ones(1,curr_layer.winSize, class(gather(input2)))/curr_layer.winSize, 'valid');
%         output = SCM(:, 1:curr_layer.winShift:end);
%         output = permute(reshape(output, nCh^2*nBin, N, size(output, 2)), [1 3 2]);
        
        prev_mask = input_layer.validFrameMask;
        output = zeros(nCh^2*nBin, nf, N, 'like', XX2);
        for i=1:N
            idx = find(prev_mask(:,i) == 0, 1, 'last');
            idx2 = fix((idx-curr_layer.winSize+curr_layer.winShift)/curr_layer.winShift);
            XX3 = squeeze(XX2(:,1:idx,i));
            SCM = conv2(XX3, ones(1,curr_layer.winSize, class(gather(input2)))/curr_layer.winSize, 'valid');
            output(:, 1:idx2, i) = SCM(:, 1:curr_layer.winShift:end);
            mask(idx2+1:end, i) = 1;
        end
        
    end
end

end
