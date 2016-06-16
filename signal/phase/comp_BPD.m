% This function compute the instantaneous frequency from phase spectrogram
% Input: 
%   phase: D x T matrix of phase spectrogram, where D is FFT-bin number and
%       T is number of frames. 
% Output: 
%   instantaneous frequency of the size size as phase. 
% The implementation is based on equation (3) of 
%   Krawczyk, Martin, and Timo Gerkmann. "STFT Phase Reconstruction in 
%   Voiced Speech for an Improved Single-Channel Speech Enhancement." (2014).
%
% Author: Xiong Xiao, Nanyang Tech. Univ., Singapore. 
% Created: 06 Jan 2015
% Last modified: 06 Jan 2015
%
function [BPD, instan_freq] = comp_BPD(phase, nFFT, frame_shift_in_sample)

phase_delta = phase(:,2:end) - phase(:,1:end-1);
phase_delta = [phase(:,1) phase_delta];

omega = 2*pi*(0:(nFFT/2))/nFFT;
offset = frame_shift_in_sample*omega;

phase_delta2 = bsxfun(@minus, phase_delta, offset');

instan_freq = atan2(sin(phase_delta),cos(phase_delta));

BPD = atan2(sin(phase_delta2),cos(phase_delta2));

if 0 
    subplot(3,1,1); imagesc(phase); title('Phase'); colorbar;
    subplot(3,1,2); imagesc(phase_delta); title('Instantaneous freq');colorbar;
    subplot(3,1,3); imagesc(BPD); title('BPD');colorbar;
end

end
