function [output] = F_beamforming_freeWeight(input_layer, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);      

W = curr_layer.W;   % W is a 2*nCh x nBin matrix. The first half store the real parts, the second half store the imaginary parts
[D2,nBin] = size(W);
nCh = D2/2;
Wcomplex = W(1:nCh,:) + W(nCh+1:end,:)*sqrt(-1);

input2 = reshape(input, nBin, nCh, T, N);

output = bsxfun(@times, input2, Wcomplex');

output = squeeze(sum(output,2));

end
