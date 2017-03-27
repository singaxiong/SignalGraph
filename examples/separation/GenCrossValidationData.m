function GenCrossValidationData()
SPR = 0;

tag = sprintf('.SPR%d', SPR);

para.local.clean_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\dev-clean', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'}); 
para.local.clean_wav_ext = 'flac';
para.local.rir_wav_root = ChoosePath4OS({'D:\Data\NoiseData\ReverbEstimation\RIR_T60_1ch', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'}); 
para.local.rir_wav_ext = 'wav';
para.local.noise_wav_root = ChoosePath4OS({'D:\Data\NoiseData\musan\noise', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'}); 
para.local.noise_wav_ext = 'wav';
para.local.useFileName = 1;      % if set to 0, load all training data to memory. otherwise, only load file names.
para.local.parallel_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\dev-separation', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'}); 
para.local.parallel_wav_root = [para.local.parallel_wav_root tag];
para.singlePrecision = 1;

fs = 16000;
para.IO.DynamicMixture.fs = fs;
para.IO.DynamicMixture.nUtt4Iteration = 100;   % dynamically generate 10000 distorted sentences for each training iteration
para.IO.DynamicMixture.SPR_PDF = 'uniform';      % distribution of the power ratio between the two sources, defined as 10log10(P1/P2), 
                                                 % where P1>P2 are the power of the two mixture signals
para.IO.DynamicMixture.SPR_para = [-1 1] * SPR;        % parameters of SPR PDF. If use uniform, it is the lowest and highest SPR allowed. 
                                                 % if use normal, it is the mean and variance of SPR. 
para.IO.DynamicMixture.addReverb = 0;            % whether to add reverberation to the mixed signal
para.IO.DynamicMixture.addNoise = 0;             % whether to add noise to the mixed signal
para.IO.DynamicMixture.SNR_PDF = 'uniform';      % distribution of SNR, choose [uniform|normal]. SNR is defined as the energy ratio P1/P2, 
                                                 % where P1>P2 are the power of the two mixture signals
para.IO.DynamicMixture.SNR_para = [0 20];       % parameters of SNR PDF. If use uniform, it is the lowest and highest SNR allowed. 
                                                 % if use normal, it is the mean and variance of SNR. 
                                                 
[base_data, para] = LoadWavRIRNoise_Libri(para, 1);
para.IO.DynamicMixture.inputFeature = para.IO.DynamicDistortion.inputFeature;
para.IO.DynamicMixture.fileReader = para.IO.DynamicDistortion.fileReader;

[data, para] = GenDynamicMixture(base_data, para);

distorted = data(1).data;
clean1 = data(2).data;
clean2 = data(3).data;

my_mkdir([para.local.parallel_wav_root '/mixed']);
my_mkdir([para.local.parallel_wav_root '/clean1']);
my_mkdir([para.local.parallel_wav_root '/clean2']);
for i=1:length(distorted)
    PrintProgress(i, length(distorted), 100);
    curr_distorted = double(distorted{i});
    curr_distorted = curr_distorted / max(abs(curr_distorted));
    distorted_filename = [para.local.parallel_wav_root '/mixed/' sprintf('utt%d.wav', i)];
    audiowrite(distorted_filename, curr_distorted, fs);

    curr_clean = double(clean1{i});
    curr_clean = curr_clean / max(abs(curr_clean));
    clean_filename = [para.local.parallel_wav_root '/clean1/' sprintf('utt%d.wav', i)];
    audiowrite(clean_filename, curr_clean, fs);

    curr_clean = double(clean2{i});
    curr_clean = curr_clean / max(abs(curr_clean));
    clean_filename = [para.local.parallel_wav_root '/clean2/' sprintf('utt%d.wav', i)];
    audiowrite(clean_filename, curr_clean, fs);
end
