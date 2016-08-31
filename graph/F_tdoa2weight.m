

function output = F_tdoa2weight(input, freq_bin)
% assume input is an array of time delay of C microphone channels. 
% freq_bin is an array of center frequencies of N FFT bins. 
delay = [0; input]; 
j = sqrt(-1);

output = exp(-j*freq_bin'*delay') / length(delay);
% output = freq_bin'*delay'/length(delay);
end
