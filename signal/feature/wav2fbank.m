% Calculate the log mel filter bank from waveform
% Author: Xiao Xiong
% Created: 4 Feb 2005
% Last modified: 4 Feb 2005
 
function [fbank] = wav2fbank(x, fs, frame_shift, nMel)
if nargin < 2
    fs = 8000;
end
if nargin < 3
    frame_shift = 0.01;
end
if nargin<4
    nMel = 23;
end

absX = wav2abs(x,fs,frame_shift);
fbank = log (abs2Mel(absX, fs, nMel));
