% This function apply RIR to the clean speech signal, and also optionally
% add additive noise at a specified SNR.
%%%%
function [y]=ApplyConstRirNoise(x,fs,RIR,NOISE,SNRdB)

% calculate direct+early reflection signal for calculating SNR
[~,delay]=max(RIR(:,1));
delay = gather(delay);
before_impulse=floor(fs*0.001);
after_impulse=floor(fs*0.05);
RIR_direct=RIR(delay-before_impulse:delay+after_impulse,1);
direct_signal=fconv(x,RIR_direct);

% obtain reverberant speech
for ch=1:size(RIR,2)
    rev_y(:,ch)=freq_conv(x,RIR(:,ch));
end

% normalize noise data according to the prefixed SNR value
if ~isempty(NOISE)
    while 1     % this is to ensure that the noise is longer than the signal
        if size(NOISE,1)<size(rev_y,1)
            NOISE = [NOISE; NOISE];
        else
            break;
        end
    end
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

y = y(delay:end,:);     % remove the delay due to RIR so the reverberant and clean speech are aligned.

end
