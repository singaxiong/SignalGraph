function GenCrossValidationData()

para.local.clean_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\dev-clean', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'}); 
para.local.clean_wav_ext = 'flac';
para.local.rir_wav_root = ChoosePath4OS({'D:\Data\NoiseData\ReverbEstimation\RIR_T60_1ch', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'}); 
para.local.rir_wav_ext = 'wav';
para.local.noise_wav_root = ChoosePath4OS({'D:\Data\NoiseData\musan\noise', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'}); 
para.local.noise_wav_ext = 'wav';
para.local.useFileName = 1;      % if set to 0, load all training data to memory. otherwise, only load file names.
para.local.parallel_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\dev-parallel', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'}); 

fs = 16000;
para.IO.DynamicDistortion.fs = fs;
para.IO.DynamicDistortion.nUtt4Iteration = 2703;   % dynamically generate 10000 distorted sentences for each training iteration
para.IO.DynamicDistortion.nHours4Iteration = 10;    % dynamically generate 10 hours of distorted speech for each training iteration
para.IO.DynamicDistortion.SNR_PDF = 'uniform';      % distribution of SNR, choose [uniform|normal]
para.IO.DynamicDistortion.SNR_para = [-20 30];      % parameters of SNR PDF. If use uniform, it is the lowest and highest SNR allowed. 
                                                    % if use normal, it is the mean and variance of SNR. 
[base_data, para] = LoadWavRIRNoise_Libri(para, 1);

[data, para] = GenDynamicDistortion(base_data, para);

distorted = data(1).data;
clean = data(2).data;

my_mkdir([para.local.parallel_wav_root '/distorted']);
my_mkdir([para.local.parallel_wav_root '/clean']);
for i=1:length(distorted)
    PrintProgress(i, length(distorted), 100);
    curr_distorted = double(distorted{i});
    curr_distorted = curr_distorted / max(abs(curr_distorted));
    distorted_filename = [para.local.parallel_wav_root '/distorted/' sprintf('utt%d.wav', i)];
    audiowrite(distorted_filename, curr_distorted, fs);

    curr_clean = double(clean{i});
    curr_clean = curr_clean / max(abs(curr_clean));
    clean_filename = [para.local.parallel_wav_root '/clean/' sprintf('utt%d.wav', i)];
    audiowrite(clean_filename, curr_clean, fs);
end
