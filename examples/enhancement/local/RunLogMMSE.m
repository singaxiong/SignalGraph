function [enhancedMag, enhancedWav] = RunLogMMSE(noisyFile, fs, isFile)
addpath('../../../../Enhancement/Loizou/MATLAB_code/statistical_based');

w = warning ('off','all');
if isFile
    noisyFile = ExtractWordsFromString_v2(noisyFile);
    noisyFile = noisyFile{1};
    % [enhancedWav, fs] = logmmse_vectorize(noisyFile);
    [enhancedWav, fs] = logmmse(noisyFile, [], isFile);
else
    [enhancedWav] = logmmse(noisyFile, fs, isFile);
end

if fs==16000
    fftlen = 512;
else
    fftlen = 256;
end

[~,enhancedMag] = wav2abs_multi(enhancedWav, fs, 0.01, 0.025, fftlen);
enhancedMag = 2*log(abs(enhancedMag(1:fftlen/2+1,:)));

end
