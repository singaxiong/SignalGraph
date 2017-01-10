% This script test an LSTM based speech mask predictor, using simulated or
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
layer = lstm.layer;
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

para.NET.nSequencePerMinibatch = 1; % for every minibath we only have one utterance;


%load test file list
etdt = {'et05', 'dt05'};
mode = {'simu'};
places = {'str', 'ped', 'caf', 'bus'};


para.out_layer_idx = length(layer) + [-2]; 
para.IO.isTensor = [0 0];

for e= 1:length(etdt)
    for m = 1:length(mode)
        for p =  1:length(places)
            
            dir = ['wav/' etdt{e} '_' places{p} '_' mode{m} '/'];
            omlsadir = ['omlsa_mask/' etdt{e} '_' places{p} '_' mode{m} '/'];
            src_dir = ['F:\Chime\data\audio\16kHz\isolated\'  etdt{e} '_' places{p} '_' mode{m} '/'];
            mkdir(dir);
            mkdir(omlsadir);
            filelists = findFiles(src_dir, 'wav');
            for i = 1:length(filelists)
                file = filelists{i};
                %if ~isempty(regexpi(file, 'CH6'))
                    Data_test(1).data{1} = file;
                    [wav, Fs] = audioread(file);
                    
                    output = FeatureTree2(Data_test, para, layer);
                    para.out_layer_idx = 2;
                    NoiseX = FeatureTree2(Data_test, para, layer(1:2));
                    EnhanceX = NoiseX{1}{1} .* output{1}{1};
                    
                    omlsaX = omlsa(wav, Fs, output{1}{1});
                    omlsax = OverlapAdd2(abs(omlsaX), angle(NoiseX{1}{1}), 400, 160);
                    enhanced = OverlapAdd2(abs(EnhanceX), angle(NoiseX{1}{1}), 400, 160);
                    words = ExtractWordsFromString_v2(file, '/');
                    audiowrite([dir words{end}], enhanced/max(abs(enhanced)), 16000);
                    audiowrite([omlsadir words{end}], omlsax/max(abs(omlsax)), 16000);
                %end
                
            end
        end
    end
end

