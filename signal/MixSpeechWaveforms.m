function [mixed, component1, component2] = MixSpeechWaveforms(wav1, wav2, SPR)

wav1 = wav1(:)';
n1 = length(wav1);
wav2 = wav2(:)';
n2 = length(wav2);

% scale wav2 to obtain desired SPR, the signal power ratio in dB
power1 = mean(wav1.^2);
power2 = mean(wav2.^2);
scale2 = sqrt(power1/power2 * 10^(-SPR/10));
if scale2<1
    wav2 = scale2*wav2;
else
    wav1 = wav1/scale2;
end

% if the two waveforms have different length, repeat the shorter one to
% match the longer one

if n1>n2
    nRepeat = ceil(n1/n2);
    wav2 = repmat(wav2, 1, nRepeat);
    wav2(n1+1:end) = [];
else
    nRepeat = ceil(n2/n1);
    wav1 = repmat(wav1, 1, nRepeat);    
    wav1(n2+1:end) = [];
end

component1 = wav1;
component2 = wav2;
mixed = component1 + component2;

end
