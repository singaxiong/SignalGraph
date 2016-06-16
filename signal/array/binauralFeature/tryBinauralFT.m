function tryBinauralFT

audio_root = 'F:\Data\CHiME3\isolated\dt05_bus_real';
audio_files = findFiles(audio_root, 'wav');

wav_file = audio_files{1};
idx = regexp(wav_file, 'CH');
wav_file_prefix = wav_file(1:idx(end)+1);
for i=1:6
    [wav(:,i), fs] = wavread([wav_file_prefix num2str(i) '.wav']);
end
% [wav1,fs] = wavread('c0bc020f_29.3dB_174_ch1.wav');
% [wav5,fs] = wavread('c0bc020f_29.3dB_174_ch5.wav');

frameLength = 1024;
overlap = 0.5;
micDist = 0.2;

[tauFull, ampFull, td, ld] = binaural_ft(wav(:,[1 2]),fs,frameLength,overlap,micDist);

tau_vec = mat2vec(tauFull);
tau_vec = max(-0.002, tau_vec);
tau_vec = min(0.002, tau_vec);
hist(tau_vec,1000)
figure; imagesc(log(ampFull))
figure; imagesc(tauFull);

spec1 = wav2abs(wav(:,1), fs);
spec2 = wav2abs(wav(:,2), fs);
imagesc(MVN(log([spec1.^2 spec2.^2]))')

para.notest = 1;
[processing, cost, recon_t, cost_t] = train_linearMapping(log(spec2'), log(spec1'), [], [], para);

end
