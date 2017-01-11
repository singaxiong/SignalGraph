% This script test an LSTM based speech mask MVDR, using simulated or
% real data of CHiME-4 challenge.
% Sun Sining, Northwestern Polytechnical University (NWPU)
% First estiblish in 6 Dec 2016; 

clear
addpath('local', '../lib');
addpath(genpath('F:/SignalGraph'));
addpath('E:\ssn\code\voicebox');
play_show = 1; 
lstm = load('nnet/LSTM_Mask1chOfficial.U8395.771-512-512-257.L2_3E-4.LR_3E-2/nnet.itr8.LR2.4E-4.CV10.487.mat')

para = lstm.para;
nCh = 2;
para.topology.nCh=nCh;
layer = lstm.layer;
layer{1}.dim = [nCh nCh];
layer{2}.dim = [nCh*257, nCh];
layer{3}.dim = [nCh*257, nCh*257];
layer{4}.dim = [257, nCh*257];
layer = layer(1:end-2);
layer{end}.next = 1;
scale = 1e4;        % we hard code the scale to be a constant so that all network will use the same number
scale = scale/2^16; % note that we are using int16 to store waveform samples, so need to scale down
layer = InitWavScaleLayer(layer, scale);

%add MVDR beamforming layers
layer{end+1}.name = 'SpatialCovMask';
stft_idx = ReturnLayerIdxByName(layer, 'stft');
layer{end}.prev = [-1 stft_idx(1)+1-length(layer)];
layer{end}.dim = [2*para.topology.nCh^2 1]*para.topology.nFreqBin;

para.topology.BfNetType = 'MVDR_SpatialCov';
switch para.topology.BfNetType
    case 'MVDR_SpatialCov'
        layer{end+1}.name = 'MVDR_SpatialCov';
        layer{end}.prev = -1;
        layer{end}.fs = para.topology.fs;
        layer{end}.freqBin = para.topology.freqBin;
        layer{end}.dim = [para.topology.nCh 2*para.topology.nCh^2]*para.topology.nFreqBin;
        
    case 'MVDR_EigenVector'
        layer{end+1}.name = 'MVDR_EigenVector';
        layer{end}.prev = -1;
        layer{end}.fs = para.topology.fs;
        layer{end}.freqBin = para.topology.freqBin;
        layer{end}.dim = [para.topology.nCh 2*para.topology.nCh^2]*para.topology.nFreqBin;
    case 'DNN'  % we can also predict filter weights from covariance matrix. Never tried. May have bugs. 
        layerBF = genNetworkFeedForward_v2(layer{end}.dim(1), para.hiddenLayerSizeBF, BFWeightDim, 'mse', 'tanh');
    case 'LSTM'
        % to be implemented
end

%  generate the BF output

layer{end+1}.name = 'Beamforming';
stft_idx = ReturnLayerIdxByName(layer, 'stft');
layer{end}.prev = [-1 stft_idx+1-length(layer)];
layer{end}.freqBin = para.topology.freqBin;
layer{end}.dim = [1 para.topology.nCh] * para.topology.nFreqBin;


para.topology.useFileName = 1; % we use test file name as input
para.useGPU = 0;
if para.topology.useFileName
    para.IO.inputFeature = [0 1];
    para.IO.fileReader(1).name = 'wavfile';
    para.IO.fileReader(1).multiArrayFiles = 1;
    para.IO.fileReader(1).array = 0;
    para.IO.fileReader(1).fs = 16000;
    para.IO.fileReader(1).precision = 'int16';
    para.IO.fileReader(2).name = '';
else
    para.IO.inputFeature = [1 1];
    
end
para.IO.fileReader(1).multiArrayFiles = 1;
para.IO.fileReader(1).array = 1;

para.NET.nSequencePerMinibatch = 1; % for every minibath we only have one utterance;


%load test file list
etdt = {'dt05'};
mode = {'real'};%{'simu','real'};
places = {'bus'};%{'str', 'ped', 'caf', 'bus'};


para.out_layer_idx = length(layer); 
para.IO.isTensor = [0 0];

for e= 1:length(etdt)
    for m = 1:length(mode)
        for p =  1:length(places)
            
            dir = ['wav/LSTM_Mask_MVDR_specialCov_2ch/' etdt{e} '_' places{p} '_' mode{m} '/'];
            if nCh == 6
                src_dir = ['F:\Chime\data\audio\16kHz\isolated\'  etdt{e} '_' places{p} '_' mode{m} '/'];
            elseif nCh == 2
                src_dir = ['F:\Chime\data\audio\16kHz\isolated_2ch_track\'  etdt{e} '_' places{p} '_' mode{m} '/'];
            end
            mkdir(dir);
            filelists = findFiles(src_dir, 'wav');
            filelists = sort(filelists);
            for i = 1:2:length(filelists)
                file = filelists{i};
                words = ExtractWordsFromString_v2(file, '/');
                parts = file(1:end-5);
                files{1} = filelists{i};
                files{2} = filelists{i+1};
                
                Data_test(1).data{1} = files;

                output = FeatureTree2(Data_test, para, layer);
                para.out_layer_idx = 4;
                NoiseX = FeatureTree2(Data_test, para, layer(1:4));

                enhanced = OverlapAdd2(abs(output{1, 1}{1}/scale), angle(NoiseX{1}{1}), 400, 160);

                audiowrite([dir words{end}], enhanced/max(abs(enhanced)), 16000);
                para.out_layer_idx = length(layer); 

                
            end
        end
    end
end

