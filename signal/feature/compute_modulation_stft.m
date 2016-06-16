% This function compute the modulation spectrum of speech. The modulation
% spectrum can be derived from several types of feature trajectories: 1)
% log spectrogram; 2) log Mel filterbanks; 3) MFCC. Note that it is
% different from another definition of modulation spectrum that derived
% from the evelop of band-passed time domain signal. 
% Inputs:
%   wav: the time domain signal
%   para: a structure that controls various options in the function
% Outputs:
%   modu: the modulation spectrum (complex valued) of feature blocks
%   moduAvg: the average modulation spectrum (magnitude) of the input signal
%
% Author: Xiao Xiong
% Created: 1 Dec 2014
% Last Modified: 5 Dec 2014
%
function [modu, moduAvg] = compute_modulation_stft(wav, para)
type = para.type;
fs = para.fs;
win_size = para.win_size;
win_shift = para.win_shift;

spec = wav2abs(wav, fs);

if type == 1    % compute from spectrum
    data = log(spec);
elseif type == 2    % compute from Mel filterbanks
    data = log(abs2Mel(spec, fs));
elseif type == 3    % compute from static MFCC 
    data = fbank2mfcc(log(abs2Mel(spec, fs)));
end

% Perform optional normalization to the data
if isfield(para, 'norm')
    switch para.norm
        case 'CMN'
            data = CMN(data);
        case 'MVN'
            data = MVN(data);
        otherwise
            fprintf('Unknown normalization %s\n', para.norm);
    end
end

[nFr, dim] = size(data);

nBlockMax = ceil( (nFr-win_size)/win_shift ) + 1;
modu = []; moduAvg = [];
nBlock = 0;
for i=1:win_shift:nFr
    j = min(nFr, i+win_size-1);
    if j-i+1 < win_size/2   % if the remaining frames is not enough for half of the window size, discard it. 
        break;
    end
    nBlock = nBlock + 1;
    modu(:,:,nBlock) = fft(data(i:j,:), win_size);
    last_block_size = (j-i)+1;
end

if size(modu,3)==0
    return;
end

modu = modu(1:win_size/2+1,:,:);

moduAvg = sum(abs(modu(:,:,1:end-1)),3) + abs(modu(:,:,end)) * last_block_size/win_size;
moduAvg = moduAvg / (size(modu,3)-1+last_block_size/win_size);

end
