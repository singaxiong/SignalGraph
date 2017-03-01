% Load far talk, close talk, and frame label. 
function [Data, para] = LoadWavRIRNoise_Libri(para, step)

wavreader.name = 'wavfile';
wavreader.array = 0;
wavreader.precision = 'int16';

% load clean speech list
clean_list = findFiles(para.local.clean_wav_root, para.local.clean_wav_ext);
clean_list = clean_list(step:step:end);
wav_clean = {}; 
for si = 1:length(clean_list)
    [~,curr_uttID] = fileparts(clean_list{si});
    PrintProgress(si, length(clean_list), 1000, curr_uttID);
    if para.local.useFileName
        wav_clean{end+1} = clean_list{si};
    else    
        wav_clean{end+1} = InputReader(clean_list(si), wavreader);
    end
end

% load rir wav list
rir_list = findFiles(para.local.rir_wav_root, para.local.rir_wav_ext);
rir_list = rir_list(step:step:end);
wav_rir = {}; 
for si = 1:length(rir_list)
    [~,curr_uttID] = fileparts(rir_list{si});
    PrintProgress(si, length(rir_list), 100, curr_uttID);
    if para.local.useFileName
        wav_rir{end+1} = rir_list{si};
    else    
        wav_rir{end+1} = InputReader(rir_list(si), wavreader);
    end
end

% load noise wav list
noise_list = findFiles(para.local.noise_wav_root, para.local.noise_wav_ext);
noise_list = noise_list(step:step:end);
wav_noise = {}; 
for si = 1:length(noise_list)
    [~,curr_uttID] = fileparts(noise_list{si});
    PrintProgress(si, length(noise_list), 100, curr_uttID);
    if para.local.useFileName
        wav_noise{end+1} = noise_list{si};
    else    
        wav_noise{end+1} = InputReader(noise_list(si), wavreader);
    end
end

Data(1).data = wav_clean;
Data(2).data = wav_rir;
Data(3).data = wav_noise;

para.IO.DynamicDistortion.inputFeature = [1 1 1];
if para.local.useFileName
    para.IO.DynamicDistortion.inputFeature([1 2 3]) = 0;
    if isfield(para.IO, 'fileReader')
        para.IO = rmfield(para.IO, 'fileReader');
    end
    para.IO.DynamicDistortion.fileReader(1) = wavreader;
    para.IO.DynamicDistortion.fileReader(2) = wavreader;
    para.IO.DynamicDistortion.fileReader(3) = wavreader;
end
end
