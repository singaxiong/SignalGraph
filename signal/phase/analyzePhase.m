function analyzePhase()
addpath('phase_feature_extraction');
fs = 16000;
if 1
    clean_root = 'F:\Data\ReverbChallenge\REVERB_WSJCAM0_et\data\cln_test\secondary_microphone\si_et_2\c3e';
    reverb_root = 'F:\Data\ReverbChallenge\REVERB_WSJCAM0_et\data\far_test\secondary_microphone\si_et_2\c3e';
    dereverb_root = 'F:\Data\ReverbChallenge\WavOutDNN_bigDNN19fr\REVERBWSJCAM0\SimData_et_for_1ch_far_room3_A';
    uttID = 'c3ec020a';
    
    wav_clean = audioread([clean_root '\' uttID '.wav']);
    wav_noisy = audioread([reverb_root '\' uttID '_ch1.wav']);
%     [grp_phase_clean, cep_clean] = modified_group_delay_feature([clean_root '\' uttID '.wav'], 0.4, 0.9, 12);
%     [grp_phase_noisy, cep_noisy] = modified_group_delay_feature([reverb_root '\' uttID '_ch1.wav'], 0.4, 0.9, 12);
else
    wav_clean = wavread('E:\Workspace2\Data\AURORA4\WAV\test_clean_wv1\440_16k\440c020a.wv1');
    wav_noisy = wavread('E:\Workspace2\Data\AURORA4\WAV\test_train_wv1\440_16k\440c020a.wv1');
end

frame_shift = 100/fs;
frame_length = 0.02;
[~,FT_clean] = wav2abs(wav_clean, fs, frame_shift, frame_length);
[~,FT_noisy] = wav2abs(wav_noisy, fs, frame_shift, frame_length);
FT_clean = FT_clean(1:257,:);
FT_noisy = FT_noisy(1:257,:);

mag_clean = abs(FT_clean);
mag_noisy = abs(FT_noisy);
phase_clean = angle(FT_clean);
phase_noisy = angle(FT_noisy);

[instan_freq_clean] = comp_instan_freq(phase_clean);
[instan_freq_noisy] = comp_instan_freq(phase_noisy);

[BPD_clean, IF_clean] = comp_BPD(phase_clean, 512, frame_shift*fs);
[BPD_noisy, IF_clean] = comp_BPD(phase_noisy, 512, frame_shift*fs);

[group_delay_clean] = comp_group_delay(phase_clean);
[group_delay_noisy] = comp_group_delay(phase_noisy);

[mgd_clean, log_mgd_clean] = modified_group_delay_feature(wav_clean, fs, 1, 1);
[mgd_noisy, log_mgd_noisy] = modified_group_delay_feature(wav_noisy, fs, 1, 1);

subplot(4,2,1); imagesc(log(mag_clean));
subplot(4,2,3); imagesc(log_mgd_clean);
subplot(4,2,5); imagesc(instan_freq_clean);
subplot(4,2,7); imagesc(BPD_clean);

subplot(4,2,2); imagesc(log(mag_noisy));
subplot(4,2,4); imagesc(log_mgd_noisy);
subplot(4,2,6); imagesc(instan_freq_noisy);
subplot(4,2,8); imagesc(BPD_noisy);

end