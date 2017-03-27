
function BatchTestEnhanceNetByCategory(testSet, T60, noise, SNR, DEBUG)

addpath('local');
dnn_files{1} = 'nnet/EnhanceRegression.noCMN.DeltaByEqn.MbSize40.U28539.771-LSTM-2048-771.L2_3E-4.LR_3E-3/nnet.itr37.LR1E-5.CV2453.828.mat';
dnn_files{2} = 'nnet/EnhanceRegression.noCMN.DeltaByEqn.MbSize20.U28539.771-LSTM-2048-771.L2_3E-4.LR_1E-4/nnet.itr62.LR4.97E-8.CV2439.245.mat';
dnn_files{3} = 'nnet/EnhanceGaussian.noCMN.init1.expDecay.Decay0.999.DeltaByEqn.MbSize40.U28539.771-LSTM-2048-771.L2_0.LR_3E-3/nnet.itr26.LR3.62E-5.CV756.273.mat';

hasClean = 1;
hasVar = [0 0 1];
useMasking = [0 0 0];
measures = {'PESQ'}; %, 'FWSEQ', 'CD', 'IS', 'LLR', 'WSS'};
fs=16000;

for i=1:length(dnn_files)
    model(i) = PrepareNetwork4Enhancement(dnn_files{i}, hasClean, useMasking(i));
end

if testSet==1
    paraTmp = model(1).para;
    paraTmp.local.cv_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\dev-parallel', '/media/xiaoxiong/DATA1/data1/Libri/LibriSpeech/dev-parallel'});
    [Data_cv, paraTmp] = LoadParallelWav_Libri(paraTmp, 10);
    scoreFile = ['nnet/scores/Libri_dev-parallel.mat'];
else
    % NN is good for f16, machine gun, white
    paraTmp = model(1).para;
    paraTmp.local.cv_wav_root = ['D:\Data\NoiseData\Libri\LibriSpeech\test-clean-distorted\Sent100\T60-' num2str(T60) '_' noise '_SNR' num2str(SNR) 'dB'];
    paraTmp.local.cv_wav_root_clean = 'D:\Data\NoiseData\Libri\LibriSpeech\test-clean';
    paraTmp.local.cv_wav_clean_ext = 'flac';
    [Data_cv, paraTmp] = LoadParallelWav_Libri(paraTmp, 1);
    scoreFile = ['nnet/scores/Libri_T60-' num2str(T60) '_' noise '_SNR' num2str(SNR) 'dB.mat'];

    if testSet==3
        Data_cv(1).data = {}; Data_cv(2).data = {};
        for i=1:20
            Data_cv(1).data{i} = ['D:\Data\NoiseData\LDC2015S13_Sample\distorted\Sent1\T60-0.3_factory1_SNR10dB\LDC2015S13.wav ' num2str((i-1)*5), ' ' num2str(i*5)];
            Data_cv(2).data{i} = ['D:\Data\NoiseData\LDC2015S13_Sample\clean\LDC2015S13.wav ' num2str((i-1)*5), ' ' num2str(i*5)];
        end
        scoreFile = ['nnet/scores/LDC2015S13_Sample_T60-' num2str(T60) '_' noise '_SNR' num2str(SNR) 'dB.mat'];
    end
end
for i=1:length(model)
    model(i).para.IO = paraTmp.IO;
end

for i = 1:1:length(Data_cv(1).data)
    noisyFile{i} = Data_cv(1).data{i};
    words = ExtractWordsFromString_v2(noisyFile{i});
    if str2num(words{3}) - str2num(words{2}) > 8; continue; end     % skip long sentences for speed
    
    fprintf('%s - %s\n', dos2unix(noisyFile{i}), datestr(now));

    DataTmp(1).data{1} = noisyFile{i};
    if hasClean
        DataTmp(2).data{1} = Data_cv(2).data{i};
    end
    
    for j = 1:length(model)
        [noisy{j}, enhanced{j}, clean{j}, enhancedWav{j}, mask{j}, variance{j}] = ...
            RunEnhanceNN(DataTmp, model(j).layer, model(j).para);
    end
    [enhanced{length(model)+1}, enhancedWav{length(model)+1}] = RunLogMMSE(DataTmp(1).data{1});
    
    noisyWav = double(InputReader(DataTmp(1).data{1}, model(1).para.IO.fileReader(1)));
    if hasClean
        cleanWav = double(InputReader(DataTmp(2).data{1}, model(1).para.IO.fileReader(2)));
        allWav = enhancedWav;
        allWav{end+1} = noisyWav';
        [scores] = RunObjectiveMeasures(cleanWav, allWav, measures, fs, DEBUG);
    end
    
    
    if DEBUG
        nFr = cell2mat(cellfun(@size, enhanced, 'UniformOutput', 0)');
        nFr = min(nFr(:,2));
        dim = 257;
        clear displayMatrix
        displayMatrix{1} = double(noisy{1}(1:dim, 1:nFr));
        for j=1:length(enhanced)
            displayMatrix{end+1} = double(enhanced{j}(1:dim, 1:nFr));
        end
        displayMatrix{end+1} = double(clean{1}(1:dim, 1:nFr));
        
        figure(1); imagesc(CMN(cell2mat(displayMatrix')')'); title(i); pause(0.01);
        
%         frame_idx = round(nFr/2);
%         
%         figure(2);
%         for kk = frame_idx
%             subplot(2,1,1);
%             for ii = [1 3]
%                 plot(enhancedLogSpec{ii}(1:257,kk));hold on
%             end
%             legend('1',  '3'); hold off;
%             title(kk);
%             subplot(2,1,2);
%             for ii = [1 3]
%                 if hasVar(ii)
%                     plot(MVN(enhancedLogSpec{ii}(1:257,kk)));hold on
%                     plot(MVN(sqrt(variance{ii}(1:257,kk))));hold on
%                 end
%             end
%             legend('1',  '3'); hold off;
%             pause(1)
%         end
        
        if DEBUG>1
            playDur = min(length(enhancedWav{end}), 16000*5);
            soundsc(noisyWav(1:playDur), 16000); pause
            for j=1:length(enhancedWav)
                soundsc(enhancedWav{j}(1:playDur), 16000);    pause
            end
        end
    end
end
save(scoreFile, 'scores', 'dnn_files', 'measures', 'noisyFile');
end
