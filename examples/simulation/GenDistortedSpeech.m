function GenDistortedSpeech()

[wav_clean,fs] = audioread('audio/19-198-0007.flac'); % a clean sentence taking from Libri corpus train set
wav_noise = audioread('audio/noise-sound-bible-0003.wav'); % a noise signal taking from MUSAN corpus

useGPU = 1; % whether to use GPU
% the following parameters can be randomly sampled from a distribution
SNR = 10;   % signal-to-noise ratio in dB
t60 = 0.5;  % reverberation T60 time measure how long it takes for the reverberation to decreases its power by 60dB 

RIR = genMultiChannelRIR(t60, useGPU)';    % generate the room impulse responses

[distorted, reverb_only] = ApplyConstRirNoise(wav_clean, fs, RIR, wav_noise, SNR, useGPU);
distorted = gather(distorted);      % move data from GPU to CPU memory
reverb_only = gather(reverb_only);

% you can listen to the signals
soundsc(wav_clean, fs);
soundsc(wav_noise, fs);
soundsc(distorted(:,1), fs);
soundsc(reverb_only(:,1), fs);

% or see the spectrogram of the signals
[~,spec_clean] = wav2abs_multi(wav_clean, fs);
[~,spec_noise] = wav2abs_multi(wav_noise, fs);
[~,spec_reverb_only] = wav2abs_multi(reverb_only, fs);
[~,spec_distorted] = wav2abs_multi(distorted, fs);
nBin = 257;
spec_clean = squeeze(log(abs(spec_clean(1:nBin,:,:))));
spec_noise = squeeze(log(abs(spec_noise(1:nBin,:,:))));
spec_reverb_only = reshape(log(abs(spec_reverb_only(1:nBin,:,:))), nBin*8, size(spec_reverb_only,3));
spec_distorted = reshape(log(abs(spec_distorted(1:nBin,:,:))), nBin*8, size(spec_distorted,3));

figure(1); subplot(2,1,1); imagesc(spec_clean); colorbar; title('clean');
subplot(2,1,2); imagesc(spec_noise); colorbar; title('noise');
figure(2); imagesc(spec_reverb_only); colorbar; title('reverb only');
figure(3); imagesc(spec_distorted); colorbar; title('reverb and noise');

end
