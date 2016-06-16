
function output = F_power_spectrum(input)
% assume the input is a DxT matrix of complex spectrum, where D is the dimension of the feature vector, 
% T is the number of frames in the minibatch or utterance
output = abs(input.*conj(input));
end
