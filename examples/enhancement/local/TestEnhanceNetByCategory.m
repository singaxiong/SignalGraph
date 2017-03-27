
function TestEnhanceNetByCategory(dnn_files, testSet, T60, noise, SNR, IsMaskNet, measures, useGPU, DEBUG)

addpath('local');
% dnn_files{1} = 'nnet/EnhanceRegression.noCMN.DeltaByEqn.MbSize40.U28539.771-LSTM-2048-771.L2_3E-4.LR_3E-3/nnet.itr37.LR1E-5.CV2453.828.mat';
% dnn_files{2} = 'nnet/EnhanceRegression.noCMN.DeltaByEqn.MbSize20.U28539.771-LSTM-2048-771.L2_3E-4.LR_1E-4/nnet.itr62.LR4.97E-8.CV2439.245.mat';
% dnn_files{3} = 'nnet/EnhanceGaussian.noCMN.init1.expDecay.Decay0.999.DeltaByEqn.MbSize40.U28539.771-LSTM-2048-771.L2_0.LR_3E-3/nnet.itr26.LR3.62E-5.CV756.273.mat';

hasClean = 1;
if ~exist('IsMaskNet', 'var');      IsMaskNet = [0 0 0];    end
if ~exist('measures', 'var');       measures = {'PESQ'};    end%, 'FWSEQ', 'CD', 'IS', 'LLR', 'WSS'};
if ~exist('DEBUG', 'var');          DEBUG = 1;              end
fs=16000;

for i=1:length(dnn_files)
    model(i) = PrepareNetwork4Enhancement(dnn_files{i}, hasClean, IsMaskNet(i), useGPU);
end

if testSet==1
    paraTmp = model(1).para;
    paraTmp.local.cv_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\dev-parallel', '/media/xiaoxiong/DATA1/data1/Libri/LibriSpeech/dev-parallel'});
    [Data_cv, paraTmp] = LoadParallelWav_Libri(paraTmp, 10);
    scoreFile = ['nnet/scores/Libri_dev-parallel.mat'];
else
    % NN is good for f16, machine gun, white
    paraTmp = model(1).para;
    wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech', '/media/xiaoxiong/OS/data1/G/Libri/LibriSpeech'});
    paraTmp.local.cv_wav_root = [wav_root '/test-clean-distorted/Sent100/T60-' num2str(T60) '_' noise '_SNR' num2str(SNR) 'dB'];
    paraTmp.local.cv_wav_root_clean = [wav_root '/test-clean'];
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

file_idx = 1:length(Data_cv(1).data);
if DEBUG==3
    file_idx = randperm(length(file_idx));
end
for i = file_idx
    noisyFile{i} = Data_cv(1).data{i};
    words = ExtractWordsFromString_v2(noisyFile{i});
    %if str2num(words{3}) - str2num(words{2}) > 8; continue; end     % skip long sentences for speed
    
    fprintf('%s - %s\n', dos2unix(noisyFile{i}), datestr(now));

    DataTmp(1).data{1} = noisyFile{i};
    if hasClean
        DataTmp(2).data{1} = Data_cv(2).data{i};
    end
    
    for j = 1:length(model)
        [noisy{j}, enhanced{j}, clean{j}, enhancedWav{j}, noisySTFT{j}, mask{j}, variance{j}] = ...
            RunEnhanceNN(DataTmp, model(j).layer, model(j).para);
    end
    [enhanced{length(model)+1}, enhancedWav{length(model)+1}] = RunLogMMSE(DataTmp(1).data{1},[], 1);
    % [enhanced{length(model)+2}, enhancedWav{length(model)+2}] = RunLogMMSE(enhancedWav{length(model)}, 16000, 0);
    % nSample = min(length(enhancedWav{2}), length(enhancedWav{3}));
    % enhancedWav{length(model)+2} = MVN(enhancedWav{2}(1:nSample)) + MVN(enhancedWav{3}(1:nSample));
    
    noisyWav = double(InputReader(DataTmp(1).data{1}, model(1).para.IO.fileReader(1)));
    if hasClean
        cleanWav = double(InputReader(DataTmp(2).data{1}, model(1).para.IO.fileReader(2)));
        allWav = enhancedWav;
        allWav{end+1} = noisyWav';
        [scores{i}] = RunObjectiveMeasures(cleanWav, allWav, measures, fs, DEBUG);
        
        % investigate how to use variance estimate
        if 0
            Mu = enhanced{length(model)};
            Var = variance{length(model)};
            phase = angle(noisySTFT{length(model)});

            Mu2 = max(0, exp(Mu) - 1*exp(Var/2.5));
            wavEst = abs2wav(sqrt(Mu2(1:257,:))', phase', 400, 240);
            
            OldScore = scores{i}(:,length(model));
            NewScore = RunObjectiveMeasures(cleanWav, {wavEst}, measures, fs, DEBUG);
        end
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
        for j=1:length(displayMatrix)
            displayMatrix{j} = displayMatrix{j} - mean(displayMatrix{j}(:));
        end
        
        figure(1); imagesc((cell2mat(displayMatrix')')'); title(i); pause(0.01);
        figure(2); labels = {'LSTM-MSE', 'LSTM-ML', 'Statistical'};
        for j=1:2
            subplot(1,2,j); imagesc(displayMatrix{j+1}(2:150,151:min(nFr,300))); title(sprintf('%s, PESQ=%2.2f', labels{j}, scores{i}(1,j)));
        end
        
%         figure(3);
%         subplot(2,2,1); imagesc(enhanced{length(model)}(2:150,151:min(nFr,300)));   colorbar
%         subplot(2,2,2); imagesc(sqrt(variance{length(model)}(2:150,151:min(nFr,300)))); colorbar
%         subplot(2,2,[2 4]); imagesc([sqrt(variance{length(model)}(2:150,151:min(nFr,300))); sqrt(ErrorSmooth(2:150,151:min(nFr,300)))]);     colorbar
%         subplot(2,2,3); imagesc(noisy{length(model)}(2:150,151:min(nFr,300)));     colorbar
%         
%         figure(4);
        nFrDisplay = min(nFr, 300);
        displayFrIdx = 31:nFrDisplay;
%         nPanel = length(enhanced)+4;
%         displayMatrix2 = displayMatrix([5 1 4 2 3]);
       
%         displayMatrix2b = cell2mat(displayMatrix2');
%         imagesc(displayMatrix2b(:,displayFrIdx));
       
        
        displayFreqIdx = 2:100;
        nFrDisplay = min(nFr, 350);
        displayFrIdx = 31:nFrDisplay;
        clean4display = displayMatrix{end}(displayFreqIdx, displayFrIdx);
        noisy4display = displayMatrix{1}(displayFreqIdx, displayFrIdx);
        MSE4display = displayMatrix{2}(displayFreqIdx, displayFrIdx);
        mean4display = displayMatrix{3}(displayFreqIdx, displayFrIdx);
        variance4display = sqrt(variance{length(model)}(displayFreqIdx,displayFrIdx));

        Error2 = (enhanced{1}(1:257,:) - clean{length(model)}).^2;
        ErrorSmooth2 = conv2(Error2, ones(3,3)/9, 'same');
        ErrorSmooth4display2 = sqrt(ErrorSmooth2(displayFreqIdx,displayFrIdx));

        Error = (enhanced{length(model)}(1:257,:) - clean{length(model)}).^2;
        ErrorSmooth = conv2(Error, ones(3,3)/9, 'same');
        ErrorSmooth4display = sqrt(ErrorSmooth(displayFreqIdx,displayFrIdx));
        
        
        figure(7);        
        % generated using utterance 5: D:/Data/NoiseData/Libri/LibriSpeech/test-clean-distorted/Sent100/T60-0.01_m109_SNR10dB/121-123859-0001.wav
        subplot(2,2,1); imagesc(noisy4display(:,1:100)); colorbar; title('Distorted Speech'); xlabel('Frame index'); ylabel('Frequency bin');
        subplot(2,2,2); imagesc(mean4display(:,1:100)); colorbar; title('Mean estimate'); xlabel('Frame index'); ylabel('Frequency bin');
        subplot(2,2,3); imagesc(variance4display(:,1:100)); colorbar; title('Standard deviation estimate'); xlabel('Frame index'); ylabel('Frequency bin');
        subplot(2,2,4); imagesc(min(1.5*max(max(variance4display(:,1:100))), ErrorSmooth4display2(:,1:100))); colorbar; title('Absolute Estimation Error'); xlabel('Frame index'); ylabel('Frequency bin');

        
        MSE4display(1) = min(noisy4display(:));
        mean4display(1) = min(noisy4display(:));
        figure(3)
        % generated using utterance 98 D:/Data/NoiseData/Libri/LibriSpeech/test-clean-distorted/Sent100/T60-0.01_buccaneer2_SNR10dB/908-157963-0004.wav
        subplot(2,2,1); imagesc(clean4display); title('Clean Speech'); xlabel('Frame index'); ylabel('Frequency bin');
        subplot(2,2,2); imagesc(noisy4display); title('Distorted Speech'); xlabel('Frame index'); ylabel('Frequency bin');
        subplot(2,2,3); imagesc(MSE4display); title('LSTM-MSE'); xlabel('Frame index'); ylabel('Frequency bin');
        subplot(2,2,4); imagesc(mean4display); title('LSTM-ML (Mean estimate)'); xlabel('Frame index'); ylabel('Frequency bin');

        figure(4)
        subplot(2,2,1); imagesc(variance4display); colorbar; title('LSTM-ML (Standard deviation estimate)');  xlabel('Frame index'); ylabel('Frequency bin');
        subplot(2,2,2); imagesc(min(10,ErrorSmooth4display)); colorbar; title('LSTM-ML (Absolute mean estimation error)'); xlabel('Frame index'); ylabel('Frequency bin');
        subplot(2,2,3); imagesc(min(10,ErrorSmooth4display2)); colorbar; title('LSTM-MSE (Absolute mean estimation error)'); xlabel('Frame index'); ylabel('Frequency bin');
        ErrorDiff = Error2(displayFreqIdx,displayFrIdx)-Error(displayFreqIdx,displayFrIdx);
        ErrorDiff = sign(ErrorDiff) .* sqrt(abs(ErrorDiff));
        subplot(2,2,4); imagesc(ErrorDiff); colorbar; title('LSTM-MSE error - LSTM-ML error)'); xlabel('Frame index'); ylabel('Frequency bin');

        % compare whether ML estimation has imporved the harmonic ridges
        % over MSE estimate. 
        if 0        
            figure(5);        imagesc(ErrorDiff);
            figure(6);        imagesc(mean4display);
        end
        
%         pause
       
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
        
        if DEBUG==2
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
