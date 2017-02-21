% Load far talk, close talk, and frame label. 
function [Data, para, vocab] = LoadWavFilter_Reverb(para, step, dataset, datatype, distance)
nCh = para.topology.useChannel;
wavlist = [];
% note that training data do not contain real recordings. 
% real dev and eval data do not have clean version
for type_i = 1:length(datatype)
    if strcmpi(datatype, 'real') || strcmpi(dataset, 'train')
        nRoom = 1;    else; nRoom = 3; end
    for dist_i = 1:length(distance)
        for roomID = 1:nRoom
            [tasklist, taskfile] = LoadTaskFile_Reverb([para.local.wavroot '/taskFiles'], dataset, datatype{type_i}, distance{dist_i}, nCh, roomID);
            wavlist = [wavlist Tasklist2wavlist(para.local.wavroot, tasklist, dataset, datatype{type_i})];
        end
        if strcmpi(dataset, 'train'); break; end    % For training data, far and near distances are in one list, so we can stop here.
    end
end

OracleWeight= load(['nnet/OracleDereverbFilter/Ctx0Len' num2str(para.topology.FilterTempContext) '/weight.mat']);
for i=1:length(OracleWeight.wavlist)   % build an index of clean files so we can find them quickly by utterance ID
    [~,curr_uttID] = fileparts(OracleWeight.wavlist{i}{1});
    words = ExtractWordsFromString_v2(curr_uttID, '\.');
    Weight.(['U_' words{1}]) = OracleWeight.allWeight{i};
end

wavlist = wavlist(:,step:step:end);

wavreader.name = 'wavfile';
wavreader.array = 1;
wavreader.multiArrayFiles = 1;

fs = 16000;   
frame_size = fs*0.025;
frame_shift = fs*0.01;
if para.local.useFileName==0
    seglen = para.local.seglen;
    segshift = para.local.segshift;
end

% Load data file list
wav_noisy = {};  w_optimal = {};
nUtt = size(wavlist,2);
for si = 1:nUtt
    words = ExtractWordsFromString_v2(wavlist{1,si}, '/');
    curr_uttID = words{end}(1:end-4);
    PrintProgress(si, nUtt, 100, curr_uttID);
    
    [wav] = InputReader(wavlist(:,si), wavreader);
    wav = StoreWavInt16(wav);
    
    curr_weight = single(Weight.(['U_' curr_uttID]));
    curr_weight = reshape(curr_weight, numel(curr_weight), 1);
    curr_weight = [real(curr_weight); imag(curr_weight)];
    
    % synchronize the length of label and wav
    nFr = enframe_decide_frame_number(size(wav,2), frame_size, frame_shift);
    requiredLen = DecideWavLen4XFrames(nFr, frame_size, frame_shift);
    wav(:,requiredLen+1:end) = [];
    
    if para.local.useFileName
        for i=1:nCh
            wavfileArray{i} = sprintf('%s 0 %2.3f', wavlist{i,si}, size(wav,2)/fs);
        end
        wav_noisy{end+1} = wavfileArray;
        w_optimal{end+1} = curr_weight;
    else    
        wav_seg = DivideSent2Segments(wav, (seglen-1)*frame_shift+frame_size, segshift*frame_shift, 1);
        wav_noisy = [wav_noisy wav_seg'];
        
        for i=1:length(wav_seg)
            w_optimal{end+1} = curr_weight;
        end
    end
end

Data(1).data = wav_noisy;
Data(2).data = w_optimal;

para.IO.inputFeature = [1 1];
para.IO.DataSyncSet{1} = [];
para.IO.frame_rate = [16000 0];
para.IO.isTensor = [1 1];
if para.local.useFileName
    para.IO.inputFeature([1 2]) = 0;
    wavreader.precision = 'int16';
    if isfield(para.IO, 'fileReader')
        para.IO = rmfield(para.IO, 'fileReader');
    end
    para.IO.fileReader(1) = wavreader;
    para.IO.fileReader(2) = wavreader;
    para.IO.fileReader(2).array = 0;
    para.IO.fileReader(2).multiArrayFiles = 0;
end
end
