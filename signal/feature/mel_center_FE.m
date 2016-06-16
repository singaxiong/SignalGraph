% Compute the center frequency of the mel filterbank
% Inputs:
%   start_freq:     the lowerest linear frequency included in the calculation
%   linear_samp:    sampling frequency
%   N_mel:          number of mel banks used
% Output: 
%   mel_center_freq:    the center linear frequency of each mel bank
%   bin_upper_edge:     

function [mel_center_freq, bin_upper_edge] = mel_center_FE(start_freq, linear_samp, N_mel)

% get the mel version of start frequency and linear sampling rate
start_mel = linear2mel(start_freq);
samp_mel = linear2mel(linear_samp/2);

for i=1:N_mel
    tmp_mel = (i-1)*(samp_mel-start_mel) / (N_mel+1);
    mel_center_freq(i) = mel2linear(tmp_mel+start_mel);
    tmp_mel = (i+1)*(samp_mel-start_mel) / (N_mel+1);
    bin_upper_edge(i) =  mel2linear(tmp_mel+start_mel);
end