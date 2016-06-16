

function output = F_real_imag2BFweight(input, freq_bin, online)
% assume input is an array of time delay of C microphone channels. 
% freq_bin is an array of center frequencies of N FFT bins. 
[D, T] = size(input);
N = length(freq_bin);
nCh = D/N/2;
j = sqrt(-1);

% input_mean = mean(input,2);
% realpart = reshape(input_mean(1:N*nCh), N, nCh);
% imagpart = reshape(input_mean(N*nCh+1:end), N, nCh);

if online == 0
    input = mean(input,2);
end

realpart = input(1:N*nCh,:);
imagpart = input(N*nCh+1:end,:);

output = realpart + j*imagpart;
end
