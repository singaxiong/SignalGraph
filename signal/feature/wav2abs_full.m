% comp_log_Mel receives a time domain speech signal and produces its
% log Mel filterbank coefficients. It is the Matlab counterpart of the
% feature extraction program provided with AURORA2 database. 
% Author: Xiao Xiong
% Created: 18 Jul, 2005
% Last Modified: 28 Jul, 2005
% Inputs: 
%       x   1-D time domain signal
% Outputs: 
%       hist_log_mel    the log Mel filterbank coefficients
%       hist_mel        the Mel filterbank coefficients
%       hist_abs        the spectral magnitude
%       logE            the log energy item needed in computing MFCC

% the full version of wav2abs gives full control of the parameters.
function [Abs_x] = wav2abs_full(x,FFT_length, Fs, frame_length, frame_shift)

frame_size = Fs * frame_length;    % 25ms frame
frame_shift = Fs * frame_shift;
frame_overlap = frame_size - frame_shift;
% FFT
fft_x = sfft(x,frame_size,frame_shift,Fs,FFT_length);
% get the magnitude
Abs_x = abs(fft_x(2:FFT_length/2+1,:))';