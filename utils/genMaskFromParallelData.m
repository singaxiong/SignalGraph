function [mask, SNR, power_sig, power_noisy, power_noise] = genMaskFromParallelData(clean, early_reverbed, distorted, vad, fs, useSoftMask, threshold, DEBUG)
if nargin<5
    threshold = 0;
end
if nargin<6
    DEBUG = 0;
end

cleanMaskType = 'count';    % how to estimate clean mask. [none|snr|floor|count]
PowerPercentageThreshold = 0.997;   % when estiamting clean mask, we require the speech TF bins to contains at least this much of percentange of the signal. 

if ~isempty(early_reverbed)     % if early reverbed data is provided, we use it as the target
    signal = early_reverbed;
else
    signal = clean;
end
[~,spec_sig] = wav2abs(signal, fs);
[nFFT,nFr] = size(spec_sig);
nBin = 1+nFFT/2;
power_sig = abs(spec_sig(1:nFFT/2+1,:,:)).^2;

if useSoftMask
    [~,spec_noisy] = wav2abs(distorted(:,1), fs);
    power_noisy = abs(spec_noisy(1:nFFT/2+1,:,:)).^2;
    mask_SNR = min(1,power_sig ./ power_noisy);
    power_noise = [];
    SNR = [];
else
    power_noisy = [];
    noise = distorted(:,1) - signal;
    [~,spec_noise] = wav2abs(noise,fs);
    power_noise = abs(spec_noise(1:nFFT/2+1,:,:)).^2;
    SNR = 10*log10(power_sig ./ power_noise);
    mask_SNR = logical(SNR>threshold);
end

% get a vad
if ~isempty(vad)    % the best is to use a VAD from a VAD detector
    vad_clean = single(vad>0.5);
    vad_clean = conv(vad_clean, ones(5,1), 'same')>0;
    vad_clean(nFr+1:end) = [];
elseif nFr>30       % or we use an eneragy based VAD
    power_noise_sig = mean([power_sig(:, 1:10) power_sig(:, end-10+1:end)], 2);
    energy_clean = sum(power_sig);
    energy_noise_clean = sum(power_noise_sig);
    vad_clean = energy_clean > (mean(energy_clean) .* mean(energy_noise_clean));
    vad_clean = conv(vad_clean, ones(21,1), 'same')>0;
else
    vad_clean = ones(nFr,1);
end

% AND the mask with a mask from clean only
switch lower(cleanMaskType)
    case 'none'
        mask_clean = ones(size(mask_SNR));
    case 'snr'
        SNR_clean = 10*log10(power_sig ./ power_noise_clean);
        mask_clean = logical(SNR_clean > threshold);
    case 'floor'
        alpha = 100;
        noise_floor = mean(power_sig(:,vad_clean), 2) / alpha;
        mask_clean = logical(power_sig > mean(noise_floor));
    case 'count'
        power_clean_sort = double(sort(power_sig(:), 'descend'));
        power_clean_cumsum = cumsum(power_clean_sort);
        cutoff_idx = find(power_clean_cumsum > (power_clean_cumsum(end)*PowerPercentageThreshold));
        cutoff_threshold = power_clean_sort(cutoff_idx(1));
        mask_clean = logical(power_sig > cutoff_threshold);
end
mask_clean(:,vad_clean==0) = 0;
mask = mask_SNR .* mask_clean;

if DEBUG
    [~,spec_noisy] = wav2abs(distorted(:,1), fs);
    power_noisy = abs(spec_noisy(1:nFFT/2+1,:,:)).^2;
    figure(1); 
    if useSoftMask
        imagesc(log([power_sig; power_noisy])); hold on;
    else
        imagesc(log([power_sig; power_noisy; power_noise])); hold on;
    end
    plot(vad_clean*100, 'k'); hold off
    figure(2);
    subplot(4,1,1); if useSoftMask==0; imagesc(max(-10,log(power_sig))); end
    subplot(4,1,2); imagesc(mask_SNR); title(sprintf('SNR mask, %2.2f%% speech TF bins', sum(mask_SNR(:))/sum(vad_clean>0)/nBin*100));
    subplot(4,1,3); imagesc(mask_clean); title(sprintf('Clean mask by %s, %2.2f%% speech TF bins', cleanMaskType, sum(mask_clean(:))/sum(vad_clean>0)/nBin*100));
    subplot(4,1,4); imagesc(mask); title(sprintf('Final mask, %2.2f%% speech TF bins', sum(mask(:))/sum(vad_clean>0)/nBin*100));
    pause%(.01);
end
end
