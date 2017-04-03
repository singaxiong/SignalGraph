

function output = F_tdoa2weight(input_layer, curr_layer)
% assume input is an array of time delay of C microphone channels. 
% freq_bin is an array of center frequencies of N FFT bins. 
input = input_layer.a;
freq_bin = curr_layer.freq_bin;
[D,T,N] = size(input);
nCh = D+1;
delay = [zeros(1,T); input];
delay = reshape(delay, 1, nCh, T, N);

j = sqrt(-1);
output = exp(-j * bsxfun(@times, freq_bin', delay) ) / nCh;

% output is nBin x nCh x T x N
end
