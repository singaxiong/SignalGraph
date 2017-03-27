% This function dynamically generates distorted speech signals. It is
% usually used for speech enhancement or robust ASR tasks. 
%
% Authors: Xiong Xiao, NTU, Singapore
% Last Modified: 28 Feb 2017
%
function [data, para] = GenDynamicMixture(base_data, para)
paraDM = para.IO.DynamicMixture;

clean_data = base_data(1).data;

if isfield(para.IO.DynamicDistortion, 'randomizeFileOrder')
    randomizeFileOrder = para.IO.DynamicDistortion.randomizeFileOrder;
else
    randomizeFileOrder = 1;
end
[clean_data_loaded, clean_idx] = LoadOriginalWavePair(clean_data, paraDM, randomizeFileOrder,1);

switch lower(paraDM.SPR_PDF)
    case 'uniform'
        SPR = rand(1,paraDM.nUtt4Iteration) * (paraDM.SPR_para(2) - paraDM.SPR_para(1)) + paraDM.SPR_para(1);
    case 'normal'
        SPR = randn(1,paraDM.nUtt4Iteration) * paraDM.SPR_para(2) + paraDM.SPR_para(1);
        SPR = max(-50, min(50, SPR));   % do not allow too low or high SNRs
end

if paraDM.addReverb
    rir_data = base_data(2).data;
    [rir_data_loaded, rir_idx] = LoadOriginalWave(rir_data, paraDM, randomizeFileOrder,2);
end

if paraDM.addNoise
    noise_data = base_data(3).data;
    [noise_data_loaded, noise_idx] = LoadOriginalWave(noise_data, paraDM, randomizeFileOrder,3);
    
    switch lower(paraDM.SNR_PDF)
        case 'uniform'
            SNR = rand(1,paraDM.nUtt4Iteration) * (paraDM.SNR_para(2) - paraDM.SNR_para(1)) + paraDM.SNR_para(1);
        case 'normal'
            SNR = randn(1,paraDM.nUtt4Iteration) * paraDM.SNR_para(2) + paraDM.SNR_para(1);
            SNR = max(-50, min(50, SNR));   % do not allow too low or high SNRs
    end
end

if isfield(paraDM, 'seglen') && paraDM.seglen>0
    doSegmentation = 1;
    seglen = paraDM.seglen;
    segshift = paraDM.segshift;
    frame_len = paraDM.frame_len;
    frame_shift = paraDM.frame_shift;
else
    doSegmentation = 0;
end

distorted = {}; clean_data_aligned1 = {}; clean_data_aligned2 = {};
for i=1:size(clean_idx,1)
    if para.singlePrecision
        curr_clean_wav1 = single(clean_data_loaded{clean_idx(i,1)})';
        curr_clean_wav2 = single(clean_data_loaded{clean_idx(i,2)})';
        if paraDM.addReverb; curr_rir = single(rir_data_loaded{rir_idx(i)})'; end
        if paraDM.addNoise; curr_noise = single(noise_data_loaded{noise_idx(i)})'; end
    else
        curr_clean_wav1 = double(clean_data_loaded{clean_idx(i,1)})';
        curr_clean_wav2 = double(clean_data_loaded{clean_idx(i,2)})';
        if paraDM.addReverb; curr_rir = double(rir_data_loaded{rir_idx(i)})'; end
        if paraDM.addNoise; curr_noise = double(noise_data_loaded{noise_idx(i)})'; end
    end
    
    [mixed, component1, component2] = MixSpeechWaveforms(curr_clean_wav1, curr_clean_wav2, SPR(i));
    
    if paraDM.addReverb || paraDM.addNoise
        mixed_distorted = ApplyConstRirNoise(mixed, paraDM.fs, curr_rir, curr_noise, SNR(i), para.useGPU)';
        mixed_distorted = gather(mixed_distorted);
    else
        mixed_distorted = mixed;
    end
    
    %switch class(gather(clean_data_loaded{clean_idx(i)}(1)))
    %    case 'int16'
    %        mixed_distorted = StoreWavInt16(mixed_distorted);
    %        component1 = StoreWavInt16(component1);
    %        component2 = StoreWavInt16(component2);
    %    case 'float'
    if para.singlePrecision
            mixed_distorted = single(mixed_distorted); %/ max(abs(mixed_distorted)));
            component1 = single(component1); %/ max(abs(component1)));
            component2 = single(component2); %/ max(abs(component2)));
        % otherwise
    else
            mixed_distorted = double(mixed_distorted); %/ max(abs(mixed_distorted)));            
            component1 = double(component1); %/ max(abs(component1)));
            component2 = double(component2); %/ max(abs(component2)));
    end
    mixed_distorted(:, length(mixed)+1:end) = [];
    
    if doSegmentation
        mixed_distorted_seg = DivideSent2Segments(mixed_distorted, (seglen-1)*frame_shift+frame_len, segshift*frame_shift, 1);
        distorted = [distorted mixed_distorted_seg'];
        curr_clean_wav_seg1 = DivideSent2Segments(component1, (seglen-1)*frame_shift+frame_len, segshift*frame_shift, 1);
        clean_data_aligned1 = [clean_data_aligned1 curr_clean_wav_seg1'];
        curr_clean_wav_seg2 = DivideSent2Segments(component2, (seglen-1)*frame_shift+frame_len, segshift*frame_shift, 1);
        clean_data_aligned2 = [clean_data_aligned2 curr_clean_wav_seg2'];
    else
        distorted{i} = mixed_distorted;
        clean_data_aligned1{i} = component1;
        clean_data_aligned2{i} = component2;
    end
    
    PrintProgress(i, size(clean_idx,1), max(10, round(size(clean_idx,1)/10)), ...
            'GenDynamicDistortion->Mixing clean speech with RIR and noise');
end
data(1).data = distorted;
data(2).data = clean_data_aligned1;
data(3).data = clean_data_aligned2;

end


function [data, idx] = LoadOriginalWave(wavlist, paraDM, randomizeFileOrder, streamIdx)
if isempty(wavlist)
    return;
end
if randomizeFileOrder
    idx = randperm(length(wavlist));
else
    idx = 1:length(wavlist);
end
nRepeat = ceil(paraDM.nUtt4Iteration/length(idx));
idx = repmat(idx, 1, nRepeat);
idx(paraDM.nUtt4Iteration+1:end) = [];

if paraDM.inputFeature(streamIdx)
    data = wavlist;
else
    need2load = unique(idx);
    for i=1:length(need2load)
        PrintProgress(i, length(need2load), max(100, round(length(need2load)/10)), ...
            sprintf('GenDynamicMixture->LoadOriginalWav->Stream %d', streamIdx));
        data{need2load(i)} = InputReader(wavlist{need2load(i)}, paraDM.fileReader(streamIdx));
    end
end
end


function [data, idx_pair_all] = LoadOriginalWavePair(wavlist, paraDM, randomizeFileOrder, streamIdx)
if isempty(wavlist)
    return;
end
idx_pair_all = [];
for i=1:10
    if randomizeFileOrder
        idx = randperm(length(wavlist));
    else
        idx = 1:length(wavlist);
    end
    nRepeat = ceil(paraDM.nUtt4Iteration*2/length(idx));
    idx = repmat(idx, 1, nRepeat);
    if mod(length(idx), 2)==1
        idx(end) = [];
    end
    idx_pair = reshape(idx, length(idx)/2, 2);
    identical = idx_pair(:,1) == idx_pair(:,2);
    idx_pair(identical,:) = [];
    
    idx_pair_all = [idx_pair_all; idx_pair];
    if size(idx_pair_all,1) >=paraDM.nUtt4Iteration
        break;
    end
end
idx_pair_all(paraDM.nUtt4Iteration+1:end,:)= [];

if paraDM.inputFeature(streamIdx)
    data = wavlist;
else
    need2load = unique(idx_pair_all(:));
    for i=1:length(need2load)
        PrintProgress(i, length(need2load), max(100, round(length(need2load)/10)), ...
            sprintf('GenDynamicMixture->LoadOriginalWav->Stream %d', streamIdx));
        data{need2load(i)} = InputReader(wavlist{need2load(i)}, paraDM.fileReader(streamIdx));
    end
end
end
