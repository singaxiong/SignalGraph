% comp_log_Mel receives a time domain speech signal and produces its
% log Mel filterbank coefficients. It is the Matlab counterpart of the
% feature extraction program provided with AURORA2 database. 
% Author: Xiao Xiong
% Created: 18 Jul, 2005
% Last Modified: 28 Jul, 2005
% Inputs: 
%       x   MxN time domain signal, M is the number of samples, and N is
%       the number of channels. 
% Outputs: 

% the fine version of wav2abs have half of the normal frame_shift, i.e.,
% its frame_shift is 5ms instead of 10ms.
function [Abs_x, fft_x] = wav2abs_multi(x,Fs,frame_shift, frame_size, FFT_length, useGPU)
if exist('useGPU')==0  || length(useGPU)==0
    useGPU = 0;   
end
if exist('Fs')==0  || length(Fs)==0
    Fs = 8000;
end
if exist('frame_size') && length(frame_size)>0
    frame_size = Fs * frame_size;
else
    frame_size = Fs * 0.025;
end
if exist('frame_shift') && length(frame_shift)>0
    frame_shift = Fs * frame_shift;
else
    frame_shift = Fs * 0.01;
end
if exist('FFT_length')==0 || length(FFT_length)==0
    FFT_length = pow2(ceil(log2(frame_size)));
end

% FFT

% dither the speech signal
noise = randn(size(x))/2^32;
x = x + noise;

fft_x = sfft_multi(x,frame_size,frame_shift,FFT_length, [], 1, useGPU);
% get the magnitude
% Abs_x = abs(fft_x(2:FFT_length/2+1,:))';
Abs_x = [];