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
function [phase] = BPD2phase(BPD, nFFT, frame_shift_in_sample)

omega = 2*pi*(0:(nFFT/2))/nFFT;
offset = frame_shift_in_sample*omega;

phase_delta_recover = bsxfun(@plus, BPD, offset');
phase_delta_recover2 = atan2(sin(phase_delta_recover),cos(phase_delta_recover));

phase_recover(:,1) = phase_delta_recover2(:,1);
for i=2:size(phase_delta_recover2,2)
    phase_recover(:,i) = phase_recover(:,i-1) + phase_delta_recover2(:,i);
end
phase = atan2(sin(phase_recover),cos(phase_recover));

if 0 
    subplot(2,1,1); imagesc(BPD); title('BPD');colorbar;
    subplot(2,1,2); imagesc(phase); title('Phase'); colorbar;
end

end
