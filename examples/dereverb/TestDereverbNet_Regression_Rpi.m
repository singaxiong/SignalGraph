
function TestDereverbNet_Regression_Rpi
addpath('local');
dnn1 = load('nnet/Dereverb.noCMN.DeltaByEqn.MbSize20.U52013.771-LSTM-2048-771.L2_3E-4.LR_5E-3/nnet.itr10.LR4.39E-4.CV1630.073.mat');
dnn2 = load('nnet/DereverbMask.noCMN.DeltaByEqn.MbSize20.U52013.771-LSTM-1500-771.L2_3E-4.LR_5E-3/nnet.itr11.LR1.46E-4.CV1801.505.mat');

[layer1, para1] = PrepareProcessing(dnn1);
[layer2, para2] = PrepareProcessing(dnn2);

wavroot = 'D:\Data\RpiRecording';
wavfiles = findFiles(wavroot, 'wav');

for i=2:length(wavfiles)
    [wav, fs] = audioread(wavfiles{i});
    wav = StoreWavInt16(wav(:,2))';
    DataTmp(1).data{1} = wav(fs*10+1: min(fs*20,length(wav)) );
    
    output{1} = FeatureTree2(DataTmp, para1, layer1);
    output{2} = FeatureTree2(DataTmp, para2, layer2);
    
    for j = 1:2
        noisyLogSpec{j} = gather(output{j}{1}{1});
        enhancedLogSpec{j} = gather(output{j}{1}{2});
        noisySTFT{j} = gather(output{j}{1}{3});
        phase = angle(noisySTFT{j});
        noisy_wav = abs2wav(exp(noisyLogSpec{j}(1:257,:)/2)', phase', 400, 240);
        enhanced_wav{j} = abs2wav(exp(enhancedLogSpec{j}(1:257,:)/2)', phase', 400, 240);
    end
    imagesc(CMN([noisyLogSpec{1}; enhancedLogSpec{1}(1:257,:); enhancedLogSpec{2}(1:257,:)]')');
    title(regexprep(wavfiles{i}, '_', '\\_'));  pause(0.1);
    soundsc(noisy_wav, 16000); pause
    soundsc(enhanced_wav{1}, 16000);                pause
    soundsc(enhanced_wav{2}, 16000);                pause
end

end

function [layer, para] = PrepareProcessing(dnn)

layer = dnn.layer;
para = dnn.para;
para.topology.useFileName = 1;
para.local.seglen = 100;
para.local.segshift = 100;
para.useGPU = 0;

input_idx = ReturnLayerIdxByName(layer, 'input');
layer = layer(1:input_idx(end)-1);
para.IO = RemoveIOStream(para.IO, 2);
if isfield(para.topology, 'useCMN') && para.topology.useCMN==0
    log_idx = ReturnLayerIdxByName(layer, 'Log');
    para.out_layer_idx = [log_idx(1) length(layer)];
else
    cmn_idx = ReturnLayerIdxByName(layer, 'CMN');
    para.out_layer_idx = [cmn_idx(1) length(layer)];
end
stft_idx = ReturnLayerIdxByName(layer, 'stft');
para.out_layer_idx = [para.out_layer_idx stft_idx(1) 1];

reverb_root = ChoosePath4OS({'D:/Data/REVERB_Challenge', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'});   % you can set two paths, first for windows OS and second for Linux OS.
para.local.wavroot = [reverb_root];
para.local.wsjcam0root = ChoosePath4OS({'D:/Data/wsjcam0', '/media/xiaoxiong/OS/data1/G/wsjcam0_wav'});
para.local.wsjcam0ext = ChoosePath4OS({'wav', 'wv1'});
para.local.useFileName = 1;      % if set to 0, load all training data to memory. otherwise, only load file names.
para.local.loadLabel = 0;

end