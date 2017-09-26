function [wav, fs] = ReadAudioSeg(filename, time1, time2, fs)
if fs<=0    % not a valid fs
    info = audioinfo(filename);
    fs = info.SampleRate;
    TotalSamples = info.TotalSamples;
    idx2 = min(TotalSamples,round(time2*fs));
else
    idx2 = round(time2*fs);
end
idx1 = max(1, round(time1*fs));

[wav, fs] = audioread(filename, [idx1 idx2]);

end

