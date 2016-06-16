% Compute the center frequency of the mel filterbank

function [mel_center_freq] = mel_center(linear_samp, N_mel)

for i=1:N_mel
    tmp_mel = i*linear2mel(linear_samp/2) / (N_mel+1);
    mel_center_freq(i) = mel2linear(tmp_mel);
end