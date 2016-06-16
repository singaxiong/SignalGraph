
function output = F_power_spectrum_split(input)
% assume the input is a 2DxT matrix of real and imaginary parts of the 
% complex spectrum, where D is the number of frequency bins, 
% T is the number of frames in the minibatch or utterance
% ral and imagineary parts of the complex Fourier spectrum is concatenated.
% 
[D,T] = size(input);
D = D/2;
output = input(1:D,:).^2 + input(D+1:end,:).^2;
end
