% This function dynamically generates distorted speech signals. It is
% usually used for speech enhancement or robust ASR tasks. 
%
% Authors: Xiong Xiao, NTU, Singapore
% Last Modified: 28 Feb 2017
%
function [data, paraDD, data_log] = GenDynamicDistortion(base_data, paraDD)

clean_data = base_data(1).data;
rir_data = base_data(2).data;
noise_data = base_data(3).data;

% get settings
useSoftMask = ReturnFieldWithDefaultValue(paraDD, 'useSoftMask', 0);
randomizeFileOrder = ReturnFieldWithDefaultValue(paraDD, 'randomizeFileOrder', 1);
gainNorm = ReturnFieldWithDefaultValue(paraDD, 'gainNorm', 1);
vadStream = ReturnFieldWithDefaultValue(paraDD, 'vadStream', []);
if ~isempty(vadStream)
    vadData = base_data(vadStream);
end

[clean_data_loaded, clean_idx] = LoadOriginalWave(clean_data, paraDD, randomizeFileOrder,1);
if ~isempty(rir_data)
    [rir_data_loaded, rir_idx] = LoadOriginalWave(rir_data, paraDD, randomizeFileOrder,2);
end
if ~isempty(noise_data)
    [noise_data_loaded, noise_idx] = LoadOriginalWave(noise_data, paraDD, randomizeFileOrder,3);
end

if length(base_data)>3  % sometimes, we have other data, such as frame phone label. We just copy that to the output data. 
    hasExtraData = 0; extraData = [];
    for i=4:length(base_data)
        if ~isempty(vadStream) && i == vadStream; continue; end
        extraData(end+1).data = base_data(i).data(clean_idx);  % note that the extra data streams are binded with the clean data. 
        hasExtraData = 1;
    end
end

switch lower(paraDD.SNR_PDF)
    case 'uniform'
        SNR = rand(1,paraDD.nUtt4Iteration) * (paraDD.SNR_para(2) - paraDD.SNR_para(1)) + paraDD.SNR_para(1);
    case 'normal'
        SNR = randn(1,paraDD.nUtt4Iteration) * paraDD.SNR_para(2) + paraDD.SNR_para(1);
        SNR = max(-50, min(50, SNR));   % do not allow too low or high SNRs
end

if isfield(paraDD, 'seglen') && paraDD.seglen>0
    doSegmentation = 1;
    seglen = paraDD.seglen;
    segshift = paraDD.segshift;
    frame_len = paraDD.frame_len;
    frame_shift = paraDD.frame_shift;
else
    doSegmentation = 0;
end
if isfield(paraDD, 'outputDataIdxMask') && ~isempty(paraDD.outputDataIdxMask) && paraDD.outputDataIdxMask>0
    genMask = 1;
else
    genMask = 0;
end

if ~isfield(paraDD, 'outputDataIdxClean')   % by default, the clean speech should be in stream 2
    cleanStreamIdx = 2;
elseif ~isempty(paraDD.outputDataIdxClean) && paraDD.outputDataIdxClean>0   % otherwise, put it in the defined index
    cleanStreamIdx = paraDD.outputDataIdxClean;
else
    cleanStreamIdx = -1;
end

distorted = {}; clean_data_aligned = {}; mask = {};
for j=1:length(extraData)
    extraDataSeg(j).data = {};
end
for i=1:length(clean_idx)
    data_log{i} = [clean_data{clean_idx(i)}];
    if paraDD.singlePrecision
        curr_clean_wav = single(clean_data_loaded{clean_idx(i)})';
        if ~isempty(rir_data); curr_rir = single(rir_data_loaded{rir_idx(i)})'; else; curr_rir = []; end
        if ~isempty(noise_data); curr_noise = single(noise_data_loaded{noise_idx(i)})'; else; curr_noise = [];  end
    else
        curr_clean_wav = double(clean_data_loaded{clean_idx(i)})';
        if ~isempty(rir_data); curr_rir = double(rir_data_loaded{rir_idx(i)})'; else; curr_rir = []; end
        if ~isempty(noise_data); curr_noise = double(noise_data_loaded{noise_idx(i)})'; else; curr_noise = [];  end
    end
    if ~isempty(curr_rir)   % we want the direct sound to have the gain of 1
        curr_rir = curr_rir/max(curr_rir(:));
        data_log{i} = sprintf('%s\t%s', data_log{i}, rir_data{rir_idx(i)});
    else
        data_log{i} = sprintf('%s\tNULL', data_log{i});
    end
    if ~isempty(noise_data)
        data_log{i} = sprintf('%s\t%s', data_log{i}, noise_data{noise_idx(i)});
    else
        data_log{i} = sprintf('%s\tNULL', data_log{i});
    end
    data_log{i} = sprintf('%s\t%fdB', data_log{i}, SNR(i));
    
    [curr_distorted, curr_reverb, curr_direct] = ApplyConstRirNoise(curr_clean_wav, paraDD.fs, curr_rir, curr_noise, SNR(i), paraDD.useGPU);
    curr_distorted = gather(curr_distorted)';
    curr_distorted(:, length(curr_clean_wav)+1:end) = [];
    curr_direct = gather(curr_direct)';
    curr_direct(:, length(curr_clean_wav)+1:end) = [];
    %curr_reverb = gather(curr_reverb)';
    %curr_reverb(:, length(curr_clean_wav)+1:end) = [];
    
    if genMask
        if 0    
            [curr_mask, curr_SNR] = genMaskFromParallelData(curr_clean_wav, curr_distorted', paraDD.fs, 0);
        else    % it's better to use curr_direct, which contains the early reflection up to 50ms after the direct sound, as the clean reference. 
            if isempty(vadStream)
                [curr_mask] = genMaskFromParallelData(curr_clean_wav, curr_direct', curr_distorted', [], paraDD.fs, useSoftMask, 0);
            else
                [curr_mask] = genMaskFromParallelData(curr_clean_wav, curr_direct', curr_distorted', vadData.data{clean_idx(i)}, paraDD.fs, useSoftMask, 0);
            end
        end    
    end
    if gainNorm
        curr_distorted = curr_distorted / max(abs(curr_distorted(:))); 
    end
    switch class(gather(clean_data_loaded{clean_idx(i)}(1)))
        case 'int16'
            curr_distorted = StoreWavInt16(curr_distorted);
        case {'single', 'float'}
            curr_distorted = single(curr_distorted);
        otherwise
            curr_distorted = double(curr_distorted);            
    end
    
    if doSegmentation
        curr_distorted_seg = DivideSent2Segments(curr_distorted, (seglen-1)*frame_shift+frame_len, segshift*frame_shift, 1);
        distorted = [distorted curr_distorted_seg'];
        curr_clean_wav_seg = DivideSent2Segments(clean_data_loaded{clean_idx(i)}, (seglen-1)*frame_shift+frame_len, segshift*frame_shift, 1);
        if cleanStreamIdx>0
            clean_data_aligned = [clean_data_aligned curr_clean_wav_seg'];
        end
        if genMask
            curr_mask_seg = DivideSent2Segments(curr_mask, seglen, segshift, 1);
            mask = [mask curr_mask_seg'];
        end
        for j=1:length(extraData)
            tmpDataSeg = DivideSent2Segments(extraData(j).data{i}, seglen, segshift, 1);    % we assume that the extra data are having 100Hz frame rate
            extraDataSeg(j).data = [extraDataSeg(j).data tmpDataSeg'];
        end
    else
        distorted{i} = curr_distorted;
        if cleanStreamIdx>0; clean_data_aligned{i} = clean_data_loaded{clean_idx(i)}; end
        if genMask; mask{i} = curr_mask; end
    end
    
    PrintProgress(i, length(clean_idx), max(10, round(length(clean_idx)/10)), ...
            'GenDynamicDistortion->Mixing clean speech with RIR and noise');
end

nOutputStream = 3;
if ~isfield(paraDD, 'outputDataIdxDistorted')   % by default, the distorted speech should be in stream 1
    data(1).data = distorted;
elseif ~isempty(paraDD.outputDataIdxDistorted) && paraDD.outputDataIdxDistorted>0   % otherwise, put it in the defined index
    data(paraDD.outputDataIdxDistorted).data = distorted;
else
    nOutputStream = nOutputStream -1;
    % do not output distorted speech
end
if cleanStreamIdx>0
    data(cleanStreamIdx).data = clean_data_aligned;
else
    nOutputStream = nOutputStream -1;
    % do not output clean speech
end
if genMask
    data(paraDD.outputDataIdxMask).data = mask;
else
    nOutputStream = nOutputStream -1;
end

if hasExtraData
    if doSegmentation
        data = [data extraDataSeg];
    else
        data = [data extraData];
    end
end

end


function [data, idx] = LoadOriginalWave(wavlist, paraDD, randomizeFileOrder, streamIdx)
if isempty(wavlist)
    return;
end
if randomizeFileOrder
    idx = randperm(length(wavlist));
else
    idx = 1:length(wavlist);
end
nRepeat = ceil(paraDD.nUtt4Iteration/length(idx));
idx = repmat(idx, 1, nRepeat);
idx(paraDD.nUtt4Iteration+1:end) = [];

if paraDD.inputFeature(streamIdx)
    data = wavlist;
else
    need2load = unique(idx);
    for i=1:length(need2load)
        PrintProgress(i, length(need2load), max(100, round(length(need2load)/10)), ...
            sprintf('GenDynamicDistortion->LoadOriginalWav->Stream %d', streamIdx));
        % data{need2load(i)} = InputReader(wavlist{need2load(i)}, paraDD.fileReader(streamIdx));
        data{need2load(i)} = paraDD.fileReader{streamIdx}.read(wavlist{need2load(i)});
    end
end
end
