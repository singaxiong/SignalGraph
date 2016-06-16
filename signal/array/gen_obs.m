
%%%%
function [y]=gen_obs(x,RIR,NOISE,SNRdB)
% function to generate noisy reverberant data

% x=x';

% calculate direct+early reflection signal for calculating SNR
[val,delay]=max(RIR(:,1));
before_impulse=floor(16000*0.001);
after_impulse=floor(16000*0.05);
RIR_direct=RIR(delay-before_impulse:delay+after_impulse,1);
direct_signal=fconv(x,RIR_direct);

% obtain reverberant speech
for ch=1:size(RIR,2)
    rev_y(:,ch)=fconv(x,RIR(:,ch));
end

% normalize noise data according to the prefixed SNR value
if length(NOISE)>0
    NOISE=NOISE(1:size(rev_y,1),:);
    NOISE_ref=NOISE(:,1);
    iPn = diag(1./mean(NOISE_ref.^2,1));
    Px = diag(mean(direct_signal.^2,1));
    Msnr = sqrt(10^(-SNRdB/10)*iPn*Px);
    scaled_NOISE = NOISE*Msnr;
    y = rev_y + scaled_NOISE;
else
    y = rev_y;
end

y = y(delay:end,:);


