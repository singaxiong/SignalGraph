% This script test an LSTM based speech mask MVDR, using simulated or
% real data of CHiME-4 challenge.
% Sun Sining, Northwestern Polytechnical University (NWPU)
% First estiblish in 6 Dec 2016; 

clear
addpath('local', '../lib');
addpath(genpath('F:/SignalGraph'));
addpath('E:\ssn\code\voicebox');
play_show = 1; 
lstm = load('nnet/MaskBF2ch_split0_LSTM_MVDR_DNN.U8634_mixed_randPair.771-512-512-AM0-7_2048-1981.L2_0.LR_3E-3/nnet.itr1.LR2.42E-3.CV2.313.mat')

para = lstm.para;
layer = lstm.layer;
id_bf = ReturnLayerIdxByName(layer, 'Beamforming');
layer = layer(1:id_bf);

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
mode = {'real'};
places = {'bus'};%{'str', 'ped', 'caf', 'bus'};


para.out_layer_idx = length(layer); 
para.IO.isTensor = [0 0];

for e= 1:length(etdt)
    for m = 1:length(mode)
        for p =  1:length(places)
            
            dir = ['wav/LSTM_Mask_MVDR_spacialCov_2ch_CE/' etdt{e} '_' places{p} '_' mode{m} '/'];
            src_dir = ['F:\Chime\data\audio\16kHz\isolated_2ch_track\'  etdt{e} '_' places{p} '_' mode{m} '/'];
            mkdir(dir);
            filelists = findFiles(src_dir, 'wav');
            filelists = sort(filelists);
            for i = 1:2:length(filelists)
                file = filelists{i};
                parts = file(1:end-5);
                words = ExtractWordsFromString_v2(file, '/');                 
                files{1} = filelists{i};
                files{2} = filelists{i+1};

                Data_test(1).data{1} = files;

                output = FeatureTree2(Data_test, para, layer);
                para.out_layer_idx = 4;
                NoiseX = FeatureTree2(Data_test, para, layer(1:4));

                enhanced = OverlapAdd2(abs(output{1, 1}{1}), angle(NoiseX{1}{1}), 400, 160);

                audiowrite([dir words{end}], enhanced/max(abs(enhanced)), 16000);
                para.out_layer_idx = length(layer); 
               
                
            end
        end
    end
end

