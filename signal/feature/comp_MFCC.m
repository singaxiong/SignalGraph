% comp_MFCC computes the MFCC of the input time domain signal in the same
% way as the feature extraction program provided with the AURORA2 database
% Author: XIAO XIONG
% Created: 29 Jul, 2005
% Last modified: 29 Jul, 2005
% Input: 
%   x   1-D time domain signal
% Output:
%   mfcc    MxN MFCC coefficients, M is the number of frames and N is the 
%           dimension of the cepstral vector

function [mfcc] = comp_MFCC(x)

% calculate the log Mel filterbank coefficients
[hist_log_mel, hist_mel, hist_abs, hist_fft,logE] = comp_log_Mel(x);
% calculate the Mel-Scaled cepstral coefficients
mfcc = fbank2mfcc(hist_log_mel',0,0);
% append the log energy 
mfcc(:,14) = logE;