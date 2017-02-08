% This function generate the log Mel filterbanks features of the training
% and evaluation data. Note that the filterbanks features generated here is
% slightly different from that by other toolkits, such as Kaldi. 
% This recipe will always use the filterbank features generated in the same
% way as in this function, so we will be able to do joint training of
% feature extraction and acoustic modeling later. (e.g. you cannot use
% Kaldi fbank features in joint training unless you have Matlab version of
% the Kaldi feature extraction. 

function GenFbankFeatures(dataset)
addpath('lib');

% load the wavelist

chime_root = ChoosePath4OS({'D:/Data/CHiME4', '/home/xiaoxiong/CHiME4'});   % you can set two paths, first for windows OS and second for Linux OS. 
fbank_root = [chime_root '/fbank/plain'];

switch dataset
    case 'tr05_orig'
        wavlist = findFiles([chime_root '/audio/isolated/tr05_org']);
    case 'tr05'
        
    case 'dt05'
        
    case 'et05'
        
    otherwise
        fprintf('Unknown dataset: %s!\n', dataset);
end

[layer, para] = BuildFbankExtractionNet();

for i=1:length(wavlist)
    PrintProgress(i-1, length(wavlist), 100, 'Generate filterbank');
    words = ExtractWordsFromString_v2(wavlist{i}, '/');
    wav = audioread(wavlist{i});
    wav = StoreWavInt16(wav');
    Data(1).data{1} = wav;
    
    [fbank, layer2] = FeatureTree2(Data, para, layer);
    fbank = fbank{1}{1};
    
    fbank_file = [fbank_root '/' words{end-2} '/' words{end-1} '/' words{end}(1:end-4) '.fbank'];
    fbank_dir = fileparts(fbank_file);
    my_mkdir(fbank_dir);
    writeHTK(fbank_file, fbank', 'MFCC_0', 1);
    
end
