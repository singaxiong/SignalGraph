function [output] = F_beamforming_gainTDOA(input_layer, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);      

gain = curr_layer.W;   % W is a 2*nCh x nBin matrix. The first half store the real parts, the second half store the imaginary parts
TDOA = curr_layer.b;
[nCh,nBin] = size(gain);
freqBin = curr_layer.freqBin;

weight = F_tdoa2weight(TDOA(2:end), freqBin);
if isfield(curr_layer, 'useGain') && curr_layer.useGain==1
    weight = bsxfun(@times, weight, gain');
end

input2 = reshape(input, nBin, nCh, T, N);

output = bsxfun(@times, input2, conj(weight));

output = squeeze(sum(output,2));

end
