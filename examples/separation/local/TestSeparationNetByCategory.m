
function TestSeparationNetByCategory()

addpath('local');
% dnn_files{1} = 'nnet/SeparationRegression.CMN.DeltaByEqn.MbSize40.U28539.771-LSTM-2048-771.L2_0.LR_1E-3/nnet.itr19.LR1.29E-4.CV5039.393.mat';
dnn_files{1} = 'nnet/SepReg.CMN.DeltaByEqn.MbSize40.SPR0.seg100.U28539.771-LSTM-2048-771.L2_0.LR_1E-3/nnet.itr2.LR8.32E-4.CV5385.151.mat';
dnn_files{1} = 'nnet/SepReg.CMN.DeltaByEqn.MbSize40.SPR0.seg100.U28539.771-LSTM-2048-771.L2_0.LR_1E-3/nnet.itr50.LR7.96E-7.CV4840.695.mat';
dnn_files{1} = 'nnet/SepMask.CMN.DeltaByEqn.MbSize40.SPR0.U28539.771-LSTM-2048-771.L2_0.LR_1E-3/nnet.itr73.LR2.84E-8.CV4988.980.mat';
% dnn_files{1} = 'nnet/SepMask.noCMN.DeltaByEqn.MbSize40.SPR0.U28539.771-LSTM-2048-771.L2_0.LR_1E-3/nnet.itr63.LR1.21E-7.CV5861.556.mat';
DEBUG = 2;

hasClean = 1;
if ~exist('IsMaskNet', 'var');      IsMaskNet = [0 0 0];    end
if ~exist('measures', 'var');       measures = {'PESQ'};    end%, 'FWSEQ', 'CD', 'IS', 'LLR', 'WSS'};
if ~exist('DEBUG', 'var');          DEBUG = 1;              end
if ~exist('useGPU', 'var');         useGPU = 0;             end
fs=16000;

for i=1:length(dnn_files)
    model(i) = PrepareNetwork4Separation(dnn_files{i}, hasClean, useGPU);
end

paraTmp = model(1).para;
paraTmp.local.cv_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\dev-separation.SPR0', '/media/xiaoxiong/DATA1/data1/Libri/LibriSpeech/dev-parallel'});
[Data_cv, paraTmp] = LoadSeparationWav_Libri(paraTmp, 1);
scoreFile = ['nnet/scores/Libri_dev-parallel.mat'];

for i=1:length(model)
    model(i).para.IO = paraTmp.IO;
end

file_idx = 1:length(Data_cv(1).data);
if DEBUG==3
    file_idx = randperm(length(file_idx));
end
file_idx = [8 13 14,  30 32 34 38];

for i = file_idx
    noisyFile{i} = Data_cv(1).data{i};
    words = ExtractWordsFromString_v2(noisyFile{i});
    %if str2num(words{3}) - str2num(words{2}) > 8; continue; end     % skip long sentences for speed
    
    wav = InputReader(noisyFile{i}, model(1).para.IO.fileReader(1));
    if length(wav)>8*16000; continue; end
    
    fprintf('%s - %s\n', dos2unix(noisyFile{i}), datestr(now));
    
    DataTmp(1).data{1} = noisyFile{i};
    if hasClean
        DataTmp(2).data{1} = Data_cv(2).data{i};
        DataTmp(3).data{1} = Data_cv(3).data{i};
    end
    
    for j = 1:length(model)
        [mixture{j}, separated{j}, clean{j}, separatedWav{j}, mixtureSTFT{j}, mask{j}] = ...
            RunSeparationNN(DataTmp, model(j).layer, model(j).para);
    end
    
    mixedWav = double(InputReader(DataTmp(1).data{1}, model(1).para.IO.fileReader(1)));
    
    if DEBUG
        %nFr = cell2mat(cellfun(@size, separated, 'UniformOutput', 0)');
        %nFr = min(nFr(:,2));
        nFr = size(mixture{1},2);
        dim = 257;
        clear displayMatrix
        displayMatrix{1} = double(mixture{1}(1:dim, 1:nFr))-1;
        for j=1:length(separated)
            for k=1:2
                displayMatrix{end+1} = double(separated{j}{k}(1:dim, 1:nFr));
            end
        end
        for j=1:2
            displayMatrix{end+1} = double(clean{1}{j}(1:dim, 1:nFr));
        end
        for j=1:length(displayMatrix)
            displayMatrix{j} = displayMatrix{j} - mean(displayMatrix{j}(:));
        end
        
        figure(1); imagesc((cell2mat(displayMatrix')')'); title(i); pause(0.01);
        figure(2); imagesc(cell2mat(mask{1}'));
        
        if DEBUG>1
            playDur = min(length(separatedWav{end}{1}), 16000*5);
            soundsc(mixedWav(1:playDur), 16000); pause
            for j=1:length(separatedWav)
                for k=1:2
                    soundsc(separatedWav{j}{k}(1:playDur), 16000);    pause
                end
            end
        end
    end
end
end
