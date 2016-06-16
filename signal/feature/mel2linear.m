% convert mel frequency to linear frequency

function [linear_freq] = mel2linear(mel_freq)

% slow implementation
% for i=1:length(mel_freq)
%     linear_freq(i) = 700*( 10^(mel_freq(i)/2595)-1 );
% end

% fast implementation
linear_freq = 700*( 10.^(mel_freq/2595)-1 );