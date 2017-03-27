function GenEvalData()


% Get a subset of clean files
if 0
    nSent = 100;
    cleanFiles = findFiles('D:\Data\NoiseData\Libri\LibriSpeech\test-clean', 'flac');
    para.local.distorted_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\test-clean-distorted', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'});
else
    nSent = 1;
    cleanFiles{1} = 'D:\Data\NoiseData\LDC2015S13_Sample\clean\LDC2015S13.wav';
    para.local.distorted_wav_root = 'D:\Data\NoiseData\LDC2015S13_Sample\distorted';
end

cleanFilesSmall = filter_uttr_list(sort(cleanFiles), 'total', nSent);
para.local.clean_wav_files = cleanFilesSmall;
para.local.useFileName = 1;      % if set to 0, load all training data to memory. otherwise, only load file names.

fs = 16000;
para.IO.DynamicDistortion.fs = fs;
para.IO.DynamicDistortion.randomizeFileOrder = 0;
para.IO.DynamicDistortion.nUtt4Iteration = length(cleanFilesSmall);   % dynamically generate 10000 distorted sentences for each training iteration
para.IO.DynamicDistortion.SNR_PDF = 'uniform';      % distribution of SNR, choose [uniform|normal]

% Get RIR files
% rirFiles = findFiles('D:\Data\NoiseData\ReverbEstimation\RIR_T60_1ch', 'wav');
rirFiles{1} = 'D:\Data\NoiseData\ReverbEstimation\RIR_T60_1ch/small/far/RIR_smallRoom_far_t60_0.01_Angle_120.wav';
rirFiles{2} = 'D:\Data\NoiseData\ReverbEstimation\RIR_T60_1ch/medium/near/RIR_mediumRoom_near_t60_0.3_Angle_353.wav';
rirFiles{3} = 'D:\Data\NoiseData\ReverbEstimation\RIR_T60_1ch/medium/far/RIR_mediumRoom_far_t60_0.6_Angle_349.wav';
rirFiles{4} = 'D:\Data\NoiseData\ReverbEstimation\RIR_T60_1ch/large/far/RIR_largeRoom_far_t60_0.9_Angle_263.wav';
T60 = [0.01 0.3 0.6 0.9];

% Get noise files
noiseFiles = findFiles('D:\Data\NoiseData\Noisex92\16kHz', 'wav');

SNR = [20 10 0 -10];

for t60_i = 1:length(T60)
    for n_i = 1:length(noiseFiles)
        [~, noise_type] = fileparts(noiseFiles{n_i});
        for snr_i = 1:length(SNR)
            fprintf('Generating data for T60=%1.2fs, noise=%s, SNR=%ddB\n', T60(t60_i), noise_type, SNR(snr_i));
            para.local.rir_wav_files = rirFiles(t60_i);
            para.local.noise_wav_files = noiseFiles(n_i);
            para.IO.DynamicDistortion.SNR_para = [1 1]*SNR(snr_i);      % parameters of SNR PDF. If use uniform, it is the lowest and highest SNR allowed.
            
            % if use normal, it is the mean and variance of SNR.
            [base_data, para] = LoadWavRIRNoise_Libri(para, 1);
            
            [data, para] = GenDynamicDistortion(base_data, para);
            distorted = data(1).data;
            
            outdir = sprintf('%s/Sent%d/T60-%s_%s_SNR%ddB', para.local.distorted_wav_root, nSent, num2str(T60(t60_i)), noise_type, SNR(snr_i));
            my_mkdir(outdir);
            
            for i=1:length(distorted)
                PrintProgress(i, length(distorted), 100);
                [~, uttID] = fileparts(para.local.clean_wav_files{i});
                curr_distorted = double(distorted{i});
                curr_distorted = curr_distorted / max(abs(curr_distorted));
                distorted_filename = [outdir '/' uttID '.wav'];
                audiowrite(distorted_filename, curr_distorted, fs);
            end
        end
    end
end
end
