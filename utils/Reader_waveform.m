% Read single/multi channel waveforms
% Inputs:
%   files: a cell array of files to be read. Each cell represents one
%     sentence could be a cell and could be array of files itself if it is
%     multi-channel recording. A filename may also contains the starting
%     and ending time to be read, e.g. "abc.wav 0.2 0.5" will read the
%     segment from 0.2s to 0.5s of abc.wav.
%   reader: a structure that may contains following fields
%       fs: sampling frequency, if known in advance.
%       array: set to 1 if the sentence has multiple channels
%       multiArrayFiles: set to 1 if each channel is stored in one file.
%       useChannel: an array of integers specifying which channel(s) to be
%           returned.
%
%   Author: Xiong Xiao, Nanyang Technological University, Singapore
%   Last modified: 27 Jul 2016
%
function [all_wav, fs] = Reader_waveform(files, reader)

for i=1:length(files)
    if isfield(reader, 'array') && reader.array
        % read in multichannel waveforms
        if isfield(reader, 'multiArrayFiles') && reader.multiArrayFiles     % when the different channels are stored in different files
            for j=1:length(files{i})
                if isfield(reader, 'fs')
                    [tmp_wav, fs] = Reader_waveform_core(files{i}{j}, reader.fs);
                else
                    [tmp_wav, fs] = Reader_waveform_core(files{i}{j});
                end
                if j==1
                    wav = zeros(length(files{i}), length(tmp_wav));
                end
                wav(j,:) = tmp_wav;
            end
        else                                                                % when the different channels are stored in the same file
            if isfield(reader, 'fs')
                [wav, fs] = Reader_waveform_core(files{i}, reader.fs);
            else
                [wav, fs] = Reader_waveform_core(files{i});
            end
        end
        % optional selection of channels
        if isfield(reader, 'useChannel')
            wav = wav(reader.useChannel, :);
        end
    else
        % read in single channel waveform
        if isfield(reader, 'fs')
            [wav, fs] = Reader_waveform_core(files{i}, reader.fs);
        else
            [wav, fs] = Reader_waveform_core(files{i});
        end
    end
    all_wav{i} = wav;
end
end

%%
function [wav, fs] = Reader_waveform_core(file, fs)

words = ExtractWordsFromString_v2(file);
if length(words)==3
    filename = words{1};
    time1 = str2num(words{2});
    time2 = str2num(words{3});
    if nargin<2
        [wav, fs] = ReadAudioSeg(filename, time1, time2);
    else
        [wav, fs] = ReadAudioSeg(filename, time1, time2, fs);
    end
else
    [wav,fs] = audioread(file);
end
wav = wav';

end
