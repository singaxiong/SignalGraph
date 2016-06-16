% Calculate the log mel filter bank from waveform
% Author: Xiao Xiong
% Created: 4 Feb 2005
% Last modified: 4 Feb 2005
 
function [fbank] = wav2root_fbank(is_file_name, x, frame_shift);
if is_file_name ==1
    x = readNIST(x);
end
if nargin < 3
    fbank = (abs2Mel( wav2abs(x) )).^0.1;
else
    fbank = (abs2Mel( wav2abs(x,frame_shift) )).^0.1;
end