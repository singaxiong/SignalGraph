% This function apply RIR to the clean speech signal, and also optionally
% add additive noise at a specified SNR.
%%%%
function [y, rev_y, direct_signal2]=ApplyConstRirNoise(x,fs,RIR,NOISE,SNRdB, useGPU)
if nargin<6
    useGPU = 0;
end

% calculate direct+early reflection signal for calculating SNR
if useGPU
    x = gpuArray(x);
    RIR = gpuArray(RIR);
end

if ~isempty(RIR)
    [~,delay]=max(RIR(:,1));
    delay = gather(delay);
    before_impulse=floor(fs*0.001);
    after_impulse=floor(fs*0.05);
    
    start_idx= max(1,delay-before_impulse);
    end_idx = min(size(RIR,1),delay+after_impulse);
    
    RIR_direct=RIR(start_idx : end_idx,1);
    direct_signal=freq_conv(x,RIR_direct);
    %direct_signal2 = freq_conv(x,RIR(delay:end_idx,1)); % we will use this signal as the clean signal for computing TF mask
    direct_signal2 = freq_conv(x,RIR(1:end_idx,1));
    direct_signal2 = direct_signal2(delay:end,:);
    % obtain reverberant speech
    for ch=1:size(RIR,2)
        rev_y(:,ch)=freq_conv(x,RIR(:,ch));
    end
else
    delay = 1;
    direct_signal = x;
    direct_signal2 = x;
    rev_y = x;
end

% normalize noise data according to the prefixed SNR value
if ~isempty(NOISE)
    nRepeat = ceil( size(rev_y,1)/length(NOISE) );
    NOISE = repmat(NOISE, nRepeat,1);
    % sample a random start
    extraNoiseSamples = size(NOISE,1) - length(rev_y);
    idx = floor(rand(1)*extraNoiseSamples);
    NOISE = NOISE(idx(1)+1:idx(1)+length(rev_y),:); 
    
    if useGPU
        NOISE = gpuArray(NOISE);
    end
    NOISE_ref=NOISE(:,1);
    
    iPn = diag(1./mean(NOISE_ref.^2,1));
    Px = diag(mean(direct_signal.^2,1));
    Msnr = sqrt(10^(-SNRdB/10)*iPn*Px);
    scaled_NOISE = NOISE*Msnr;
    y = bsxfun(@plus, rev_y, scaled_NOISE);
else
    y = rev_y;
end

y = y(delay:end,:);     % remove the delay due to RIR so the reverberant and clean speech are aligned.
rev_y = rev_y(delay:end,:);

end
