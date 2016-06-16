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
function [instan_freq] = comp_instan_freq(phase)

phase_delta = phase(:,2:end) - phase(:,1:end-1);
phase_delta = [phase(:,1) phase_delta];

[dim, nFr] = size(phase_delta);

phase_delta_vec = reshape(phase_delta, dim*nFr, 1);
phase_delta_vec = get_principal_value(phase_delta_vec);
instan_freq = reshape(phase_delta_vec, dim, nFr);



if 0 
    subplot(3,1,1); imagesc(phase); title('Phase'); colorbar;
    subplot(3,1,2); imagesc(phase_delta); title('Phase delta');colorbar;
    subplot(3,1,3); imagesc(instan_freq); title('Phase delta principal value - Instantaneous frequency');colorbar;
end

end

function phase_delta_vec = get_principal_value(phase_delta_vec)
idx = find(phase_delta_vec>pi);
phase_delta_vec(idx) = phase_delta_vec(idx) - 2*pi;

idx = find(phase_delta_vec<-pi);
phase_delta_vec(idx) = phase_delta_vec(idx) + 2*pi;

end
