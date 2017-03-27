
function TestEnhanceNet
addpath('..\..\..\..\Enhancement\Loizou\MATLAB_code\objective_measures\quality');
addpath('..\..\..\..\Enhancement\Loizou\MATLAB_code\statistical_based');
hasClean = 1;

addpath('local');
dnn1 = load('nnet/EnhanceRegression.noCMN.DeltaByEqn.MbSize20.U28539.771-LSTM-2048-771.L2_3E-4.LR_1E-4/nnet.itr62.LR4.97E-8.CV2439.245.mat');
dnn2 = load('nnet/EnhanceGaussian.noCMN.init1.expDecay.Decay0.999.DeltaByEqn.MbSize40.U28539.771-LSTM-2048-771.L2_0.LR_3E-3/nnet.itr26.LR3.62E-5.CV756.273.mat');
dnn3 = load('nnet/EnhanceMask.noCMN.DeltaByEqn.MbSize20.U28539.771-LSTM-2048-771.L2_3E-4.LR_1E-4/nnet.itr52.LR1.32E-7.CV2654.785.mat');

useMasking = [0 0 1];
[layer1, para1] = PrepareProcessing(dnn1, hasClean, useMasking(1));
[layer2, para2] = PrepareProcessing(dnn2, hasClean, useMasking(2));
[layer3, para3] = PrepareProcessing(dnn3, hasClean, useMasking(3));

para1.local.useFileName = 1;
para2.local.useFileName = 1;
para3.local.useFileName = 1;
para1.useGPU = 0;
para2.useGPU = 0;
para3.useGPU = 0;
if 0
    para1.local.cv_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\dev-parallel', '/media/xiaoxiong/DATA1/data1/Libri/LibriSpeech/dev-parallel'}); 
    [Data_cv, para1] = LoadParallelWav_Libri(para1, 10);
elseif 1
    % NN is good for f16, machine gun, white
    para1.local.cv_wav_root = 'D:\Data\NoiseData\Libri\LibriSpeech\test-clean-distorted\Sent100\T60-0.01_white_SNR20dB';
    para1.local.cv_wav_root_clean = 'D:\Data\NoiseData\Libri\LibriSpeech\test-clean';
    para1.local.cv_wav_clean_ext = 'flac';
    [Data_cv, para1] = LoadParallelWav_Libri(para1, 1);

    if 0
        for i=1:20
            Data_cv(1).data{i} = ['D:\Data\NoiseData\LDC2015S13_Sample\distorted\Sent1\T60-0.3_factory1_SNR10dB\LDC2015S13.wav ' num2str((i-1)*5), ' ' num2str(i*5)];
            Data_cv(2).data{i} = ['D:\Data\NoiseData\LDC2015S13_Sample\clean\LDC2015S13.wav ' num2str((i-1)*5), ' ' num2str(i*5)];
        end
    end
end
para2.IO = para1.IO;
para3.IO = para1.IO;

for i = 1:1:length(Data_cv(1).data)
    DataTmp(1).data{1} = Data_cv(1).data{i};
    if hasClean
        DataTmp(2).data{1} = Data_cv(2).data{i};
    end
    
    noisyFile = ExtractWordsFromString_v2(DataTmp(1).data{1});
    noisyFile = noisyFile{1};
    w = warning ('off','all');
    logmmse(noisyFile, 'debug/tmp.wav')
    enhancedWavLogmmse = audioread('debug/tmp.wav');
    if length(enhancedWavLogmmse)>160000; continue; end

    output{1} = FeatureTree2(DataTmp, para1, layer1);
    output{2} = FeatureTree2(DataTmp, para2, layer2);
    output{3} = FeatureTree2(DataTmp, para3, layer3);
    
    for j = 1:3
        noisyLogSpec{j} = gather(output{j}{1}{1});
        enhancedLogSpec{j} = gather(output{j}{1}{2}(1:257,:));
        if hasClean
            cleanLogSpec{j} = gather(output{j}{1}{3});
            noisySTFT{j} = gather(output{j}{1}{4});
        else
            noisySTFT{j} = gather(output{j}{1}{3});
        end
        phase = angle(noisySTFT{j});
        noisy_wav = abs2wav(exp(noisyLogSpec{j}(1:257,:)/2)', phase', 400, 240);
        enhanced_wav{j} = abs2wav(exp(enhancedLogSpec{j}(1:257,:)/2)', phase', 400, 240);
        if useMasking(j)
            mask{j} = output{j}{1}{end};
        else
            mask{j} = [];
        end
    end

    [~,enhancedLogSpecLogMMSE] = wav2abs_multi(enhancedWavLogmmse, 16000, 0.01, 0.025,512);
    enhancedLogSpecLogMMSE = 2*log(abs(enhancedLogSpecLogMMSE(1:257,:)));
    nFr2 = size(enhancedLogSpecLogMMSE,2);
    nFr = max(nFr2, size(noisyLogSpec{1},2));
    enhancedLogSpecLogMMSE = [enhancedLogSpecLogMMSE enhancedLogSpecLogMMSE(:,end-(nFr-nFr2)+1:end)];
    
    if hasClean
        imagesc(CMN([noisyLogSpec{1}; enhancedLogSpecLogMMSE; mask{1}*10; enhancedLogSpec{1}(1:257,:); mask{2}*10; enhancedLogSpec{2}(1:257,:); enhancedLogSpec{3}(1:257,:); cleanLogSpec{1}(1:257,:)]')');
    else
        imagesc(CMN([noisyLogSpec{1}; enhancedLogSpec{1}(1:257,:); enhancedLogSpec{2}(1:257,:)]')');
    end
    title(i); pause(0.01);
    playDur = min(length(enhancedWavLogmmse), 16000*5);
%     soundsc(noisy_wav(1:playDur), 16000); pause
%     soundsc(enhancedWavLogmmse(1:playDur), 16000); pause
%     soundsc(enhanced_wav{1}(1:playDur), 16000);                pause
%     soundsc(enhanced_wav{2}(1:playDur), 16000);                pause
    
    clean_wav = double(InputReader(DataTmp(2).data{1}, para1.IO.fileReader(2)));
    
    fprintf('%s\n', dos2unix(noisyFile));
    
    pesq_scores(i,1) = pesq_fast(clean_wav, noisy_wav, 0, 16000);
    pesq_scores(i,2) = pesq_fast(clean_wav, enhancedWavLogmmse, 0, 16000);
    pesq_scores(i,3) = pesq_fast(clean_wav, enhanced_wav{1}, 0, 16000);
    pesq_scores(i,4) = pesq_fast(clean_wav, enhanced_wav{2}, 0, 16000);
    pesq_scores(i,5) = pesq_fast(clean_wav, enhanced_wav{3}, 0, 16000);
    fprintf('PESQ: '); fprintf('\t%2.2f ', pesq_scores(i,:)); fprintf('\n');
    
    fwseq_scores(i,1) = comp_fwseg_fast(clean_wav(:), noisy_wav, 0, 16000);
    fwseq_scores(i,2) = comp_fwseg_fast(clean_wav(:), enhancedWavLogmmse, 0, 16000);
    fwseq_scores(i,3) = comp_fwseg_fast(clean_wav(:), enhanced_wav{1}, 0, 16000);
    fwseq_scores(i,4) = comp_fwseg_fast(clean_wav(:), enhanced_wav{2}, 0, 16000);
    fwseq_scores(i,5) = comp_fwseg_fast(clean_wav(:), enhanced_wav{3}, 0, 16000);
    fprintf('SNR: '); fprintf('\t%2.2f ', fwseq_scores(i,:)); fprintf('\n');
    
    cep_scores(i,1) = comp_cep(clean_wav(:), noisy_wav, 0, 16000);
    cep_scores(i,2) = comp_cep(clean_wav(:), enhancedWavLogmmse, 0, 16000);
    cep_scores(i,3) = comp_cep(clean_wav(:), enhanced_wav{1}, 0, 16000);
    cep_scores(i,4) = comp_cep(clean_wav(:), enhanced_wav{2}, 0, 16000);
    cep_scores(i,5) = comp_cep(clean_wav(:), enhanced_wav{3}, 0, 16000);
    fprintf('CD: '); fprintf('\t%2.2f ', cep_scores(i,:)); fprintf('\n');
    
    is_scores(i,1) = comp_is(clean_wav(:), noisy_wav, 0, 16000);
    is_scores(i,2) = comp_is(clean_wav(:), enhancedWavLogmmse, 0, 16000);
    is_scores(i,3) = comp_is(clean_wav(:), enhanced_wav{1}, 0, 16000);
    is_scores(i,4) = comp_is(clean_wav(:), enhanced_wav{2}, 0, 16000);
    is_scores(i,5) = comp_is(clean_wav(:), enhanced_wav{3}, 0, 16000);
    fprintf('IS: '); fprintf('\t%2.2f ', is_scores(i,:)); fprintf('\n');

    llr_scores(i,1) = comp_llr(clean_wav(:), noisy_wav, 0, 16000);
    llr_scores(i,2) = comp_llr(clean_wav(:), enhancedWavLogmmse, 0, 16000);
    llr_scores(i,3) = comp_llr(clean_wav(:), enhanced_wav{1}, 0, 16000);
    llr_scores(i,4) = comp_llr(clean_wav(:), enhanced_wav{2}, 0, 16000);
    llr_scores(i,5) = comp_llr(clean_wav(:), enhanced_wav{3}, 0, 16000);
    fprintf('LLR: '); fprintf('\t%2.2f ', llr_scores(i,:)); fprintf('\n');

    wss_scores(i,1) = comp_wss(clean_wav(:), noisy_wav, 0, 16000);
    wss_scores(i,2) = comp_wss(clean_wav(:), enhancedWavLogmmse, 0, 16000);
    wss_scores(i,3) = comp_wss(clean_wav(:), enhanced_wav{1}, 0, 16000);
    wss_scores(i,4) = comp_wss(clean_wav(:), enhanced_wav{2}, 0, 16000);
    wss_scores(i,5) = comp_wss(clean_wav(:), enhanced_wav{3}, 0, 16000);
    fprintf('WSS: '); fprintf('\t%2.2f ', wss_scores(i,:)); fprintf('\n');
end
end

function [layer, para] = PrepareProcessing(dnn, hasClean, useMasking)

layer = dnn.layer;
para = dnn.para;
para.topology.useFileName = 1;
para.local.seglen = 100;
para.local.segshift = 100;
para.useGPU = 0;

input_idx = ReturnLayerIdxByName(layer, 'input');
if hasClean==0
    layer = layer(1:input_idx(end)-1);
    para.IO = RemoveIOStream(para.IO, 2);
end

log_idx = ReturnLayerIdxByName(layer, 'log');
if strcmpi(layer{end}.name, 'll_gaussian')
    para.out_layer_idx = [log_idx(1) length(layer)+layer{end}.prev(1) log_idx(2)];
elseif useMasking
    para.out_layer_idx = log_idx;
else
    para.out_layer_idx = [log_idx(1) length(layer)+layer{end}.prev(1) log_idx(2)];
end

stft_idx = ReturnLayerIdxByName(layer, 'stft');
para.out_layer_idx = [para.out_layer_idx stft_idx(1) 1];

if useMasking
    para.out_layer_idx(end+1) = ReturnLayerIdxByName(layer, 'hadamard')-1;
end
end