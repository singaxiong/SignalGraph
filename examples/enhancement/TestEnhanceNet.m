
function TestEnhanceNet
hasClean = 1;

addpath('local');
dnn1 = load('nnet/EnhanceMask.noCMN.DeltaByEqn.MbSize20.U28539.771-LSTM-1024-1024-771.L2_3E-4.LR_1E-4/nnet.itr6.LR8.15E-5.CV2854.043.mat');
dnn2 = load('nnet/EnhanceMask.noCMN.DeltaByEqn.MbSize20.U28539.771-LSTM-1024-771.L2_3E-4.LR_1E-4/nnet.itr19.LR1.47E-5.CV2747.257.mat');

[layer1, para1] = PrepareProcessing(dnn1, hasClean);
[layer2, para2] = PrepareProcessing(dnn2, hasClean);

para1.local.cv_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\dev-parallel', '/media/xiaoxiong/DATA1/data1/Libri/LibriSpeech/dev-parallel'}); 
para1.local.useFileName = 0;
[Data_cv, para1] = LoadParallelWav_Libri(para1, 10);

for i = 73:length(Data_cv(1).data)
    DataTmp(1).data{1} = Data_cv(1).data{i};
    if hasClean
        DataTmp(2).data{1} = Data_cv(2).data{i};
    end
    
    output{1} = FeatureTree2(DataTmp, para1, layer1);
    output{2} = FeatureTree2(DataTmp, para2, layer2);
    
    for j = 1:2
        noisyLogSpec{j} = gather(output{j}{1}{1});
        enhancedLogSpec{j} = gather(output{j}{1}{2});
        if hasClean
            cleanLogSpec{j} = gather(output{j}{1}{3});
            noisySTFT{j} = gather(output{j}{1}{4});
        else
            noisySTFT{j} = gather(output{j}{1}{3});
        end
        phase = angle(noisySTFT{j});
        noisy_wav = abs2wav(exp(noisyLogSpec{j}(1:257,:)/2)', phase', 400, 240);
        enhanced_wav{j} = abs2wav(exp(enhancedLogSpec{j}(1:257,:)/2)', phase', 400, 240);
        mask{j} = output{j}{1}{end};
    end
    
    if hasClean
        imagesc(CMN([noisyLogSpec{1}; mask{1}*10; enhancedLogSpec{1}(1:257,:); mask{2}*10; enhancedLogSpec{2}(1:257,:); cleanLogSpec{1}(1:257,:)]')');
    else
        imagesc(CMN([noisyLogSpec{1}; enhancedLogSpec{1}(1:257,:); enhancedLogSpec{2}(1:257,:)]')');
    end
    title(i);  pause;
    playDur = min(length(noisy_wav), 16000*5);
    soundsc(noisy_wav(1:playDur), 16000); pause
%     soundsc(enhanced_wav{1}(1:playDur), 16000);                pause
    soundsc(enhanced_wav{2}(1:playDur), 16000);                pause
end
end

function [layer, para] = PrepareProcessing(dnn, hasClean)

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
para.out_layer_idx = log_idx;

stft_idx = ReturnLayerIdxByName(layer, 'stft');
para.out_layer_idx = [para.out_layer_idx stft_idx(1) 1];

para.out_layer_idx(end+1) = ReturnLayerIdxByName(layer, 'hadamard')-1;
end