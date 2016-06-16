% mel_window generates the mel triangle window when given the number of mel
% bins and the number of linear frequency bins
% Author: Xiao Xiong
% Created: 18 Jul, 2005
% Last modified: 29 Jul, 2005
% Inputs: 
%       N_mel_bin   number of mel banks used
%       N_linear_freq_bin   number of fourier transform bins
%       linear_samp     sampling rate
% Outputs: 
%       window:  the mel window
function [window] = mel_window_FE( N_mel_bin, N_linear_freq_bin, linear_samp )
% N_linear_freq_bin is the number of frequency bins used by FFT, e.g. 256

FFT_length = N_linear_freq_bin*2;

start_freq = 64; % the frequency below 64Hz is not considered

% find the mel bin centers
[bin_lower_edge, bin_upper_edge] = mel_bank_range(start_freq, linear_samp, N_mel_bin );

% round off the mel bin centers to the linear freq bins
rounded_bin_lower_edge = round( bin_lower_edge/linear_samp*2* (N_linear_freq_bin) );
for i=1:N_mel_bin
    win_length(i) = floor(FFT_length*bin_upper_edge(i)/linear_samp+0.5) - rounded_bin_lower_edge(i) + 1;
end

le = rounded_bin_lower_edge;

window = zeros(N_linear_freq_bin, N_mel_bin);
% window for k=1:N_mel_bin
for i=1:N_mel_bin
    if i < N_mel_bin
        lower_length = le(i+1)-le(i)+1;
    else
        lower_length = le(i-1)+win_length(i-1)-le(i);
    end
    higher_length = win_length(i)-lower_length+1;
    for j = le(i) : le(i)+lower_length-1
        window(j,i) = (j+1-le(i)) / lower_length;
    end
    for j = le(i)+lower_length : le(i)+win_length(i)-1
        window(j,i) = (le(i)+win_length(i)-j) / higher_length;
    end
end  