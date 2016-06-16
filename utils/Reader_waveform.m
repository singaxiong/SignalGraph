function [wav, fs] = Reader_waveform(files, fs)

for i=1:length(files)
%     fprintf('%s\n', files{i});
    words = ExtractWordsFromString_v2(files{i});
    if length(words)==3
        filename = words{1};
        time1 = str2num(words{2});
        time2 = str2num(words{3});
        if nargin<2
            [wav{i}, fs] = ReadAudioSeg(filename, time1, time2);
        else
            [wav{i}, fs] = ReadAudioSeg(filename, time1, time2, fs);
        end
    else
        [wav{i},fs] = audioread(files{i});
    end
end
end