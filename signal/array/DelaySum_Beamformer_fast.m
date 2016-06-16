
%%%%%%@function: Delay and Sum beamformer%%%%%%%%%%%%
function [xout] = DelaySum_Beamformer_fast(sig,tde_est)
%%start to process with beamformer
nChs = size(sig,2);

%%perform Delay and Sum beamforming
lfft = 1024; %32ms window size for beamforming
ShiftP = 0.25;
Wsz = lfft;
INC = Wsz*ShiftP;
W = hann(Wsz);
%Apply the Pre-emphasis
pre_emph=0.925;
sig=filter([1 -pre_emph],1,sig);
%%segment the signals into frames
for ich = 1:nChs
    Seg_ch(:,ich,:) = enframe(sig(:,ich),W, INC).';
end
% outY = zeros(size(Seg_ch(:,:,1)));
fft_Seg_ch = fft(Seg_ch,lfft);
% hfft_Seg_ch = fft_Seg_ch(1:lfft/2,:,:);
% mi = 1:lfft/2;
mi = 1:lfft;

% hfft_Seg_ch = fft_Seg_ch;

mi = mi(:);
for ich = 1:nChs
    EV(:,ich) = exp(-1i*2*pi*mi./lfft*tde_est(ich));
end
mat_EV = repmat(EV,[1 1 size(fft_Seg_ch,3)]);
% newnFrame = size(outY,1);
mat_Y = conj(mat_EV).*fft_Seg_ch;

outY = reshape(sum(mat_Y,2),size(fft_Seg_ch,1), size(fft_Seg_ch,3));

% outY = [houtY;flipud(houtY)];
% xoutnow = ifft(outY);
xoutnow = ifft(outY);
xoutnow = xoutnow.';
xout = overlapadd(xoutnow,W,INC);
xout = real(xout);
%Undo the effect of Pre-emphasis
xout=filter(1,[1 -pre_emph],xout);
