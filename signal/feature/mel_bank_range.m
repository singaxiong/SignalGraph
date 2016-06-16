% Compute the frequency range of the mel filterbank
% Inputs:
%   start_freq:     the lowerest linear frequency included in the calculation
%   linear_samp:    sampling frequency
%   N_mel:          number of mel banks used
% Output: 
%   bin_lower_edge:    the lower edge linear frequency of each mel bank
%   bin_upper_edge:    the upper edge linear frequency of each mel bank

function [bin_lower_edge, bin_upper_edge] = mel_bank_range(start_freq, linear_samp, N_mel)

% get the mel version of start frequency and linear sampling rate
start_mel = linear2mel(start_freq);
samp_mel = linear2mel(linear_samp/2);

% slow implementation
if 0
    for i=1:N_mel
        % find the lower edge linear frequency of the bank
        tmp_mel = (i-1)*(samp_mel-start_mel) / (N_mel+1);
        bin_lower_edge(i) = mel2linear(tmp_mel+start_mel);
        % find the upper edge linear frequency of the bank
        tmp_mel = (i+1)*(samp_mel-start_mel) / (N_mel+1);
        bin_upper_edge(i) =  mel2linear(tmp_mel+start_mel);
    end
else
    % fast implementation
    i=1:N_mel;
    % find the lower edge linear frequency of the bank
    tmp_mel = (i-1)*(samp_mel-start_mel) / (N_mel+1);
    bin_lower_edge = mel2linear(tmp_mel+start_mel);
    % find the upper edge linear frequency of the bank
    tmp_mel = (i+1)*(samp_mel-start_mel) / (N_mel+1);
    bin_upper_edge =  mel2linear(tmp_mel+start_mel);
end
