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
function [all_wav] = Reader_binary(files, reader)

for i=1:length(files)
    [wav] = Reader_binary_core(files{i}, reader);
    if isfield(reader, 'precision')
        switch lower(reader.precision)
            case 'int16'
                wav = StoreWavInt16(wav);
            case 'single'
                wav = single(wav);
            case 'double'
                wav = double(wav);
        end
    end
    all_wav{i} = wav;
end
end

%%
function [data] = Reader_binary_core(file, reader)

words = ExtractWordsFromString_v2(file);

selectChannel = 0;
selectTime = 0;

filename = words{1};
if length(words)==4 || length(words)==2
    channelID = str2num(words{end});
    selectChannel = 1;
end
if length(words)>=3
    time1 = str2num(words{2});
    time2 = str2num(words{3});
    selectTime = 1;
end

big_endian = ReturnFieldWithDefaultValue(reader, 'big_endian', 0);

switch lower(reader.name)
    case 'raw'
        [data] = read08(filename, big_endian);   % you have to set fs correctly in the reader.
    otherwise
        if selectTime
            [data] = reader.reader.read(filename, [time1 time2]);
        else
            [data] = reader.reader.read(filename);
        end
end

if selectChannel
    data = data(channelID,:);
end

end
