
function [output] = F_power_spectrum(input_layer)
% assume the input is a DxTxN matrix of complex spectrum, where D is the dimension of the feature vector, 
% T is the number of frames in the minibatch or utterance, and N is the
% number of sentences
input = input_layer.a;
output = abs(input.*conj(input));

end
