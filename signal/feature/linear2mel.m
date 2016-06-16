% convert linear frequency to mel frequency

function [mel_freq] = linear2mel(linear_freq)

for i=1:length(linear_freq)
    mel_freq(i) = 2595*log10(1+linear_freq(i)/700);
end