% This function dynamically generates distorted speech signals. It is
% usually used for speech enhancement or robust ASR tasks. 
%
% Authors: Xiong Xiao, NTU, Singapore
% Last Modified: 28 Feb 2017
%
function [data, para] = GenDynamicDistortion(base_data, para)
paraDD = para.IO.DynamicDistortion;

clean_data = base_data(1).data;
rir_data = base_data(2).data;
noise_data = base_data(3).data;

[clean_data_loaded, clean_idx] = LoadOriginalWave(clean_data, paraDD, 1);
[rir_data_loaded, rir_idx] = LoadOriginalWave(rir_data, paraDD, 2);
[noise_data_loaded, noise_idx] = LoadOriginalWave(noise_data, paraDD, 3);

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

distorted = {}; clean_data_aligned = {};
for i=1:length(clean_idx)
    curr_clean_wav = double(clean_data_loaded{clean_idx(i)})';
    curr_rir = double(rir_data_loaded{rir_idx(i)})';
    curr_noise = double(noise_data_loaded{noise_idx(i)})';
    curr_distorted = ApplyConstRirNoise(curr_clean_wav, paraDD.fs, curr_rir, curr_noise, SNR(i))';
    
    switch class(gather(clean_data_loaded{clean_idx(i)}(1)))
        case 'int16'
            curr_distorted = StoreWavInt16(curr_distorted);
        case 'float'
            curr_distorted = single(curr_distorted/ max(abs(curr_distorted)));
        otherwise
            curr_distorted = double(curr_distorted/ max(abs(curr_distorted)));            
    end
    curr_distorted(:, length(curr_clean_wav)+1:end) = [];
    
    if doSegmentation
        curr_distorted_seg = DivideSent2Segments(curr_distorted, (seglen-1)*frame_shift+frame_len, segshift*frame_shift, 1);
        distorted = [distorted curr_distorted_seg'];
        curr_clean_wav_seg = DivideSent2Segments(clean_data_loaded{clean_idx(i)}, (seglen-1)*frame_shift+frame_len, segshift*frame_shift, 1);
        clean_data_aligned = [clean_data_aligned curr_clean_wav_seg'];
    else
        distorted{i} = curr_distorted;
        clean_data_aligned{i} = clean_data_loaded{clean_idx(i)};
    end
    
    PrintProgress(i, length(clean_idx), max(10, round(length(clean_idx)/10)), ...
            'GenDynamicDistortion->Mixing clean speech with RIR and noise');
end
data(1).data = distorted;
data(2).data = clean_data_aligned;

end


function [data, idx] = LoadOriginalWave(wavlist, paraDD, streamIdx)
if isempty(wavlist)
    return;
end
idx = randperm(length(wavlist));
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
        data{need2load(i)} = InputReader(wavlist{need2load(i)}, paraDD.fileReader(streamIdx));
    end
end
end
