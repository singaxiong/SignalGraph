% assume the input is a DxTxN tensor. The first D/2 dimensions are store the
% real parts, and the last D/2 dimensions store the imaginary parts. The
% function returns the corresponding complex numbers
%
function [output] = F_complex2realImag(input_layer)
input = input_layer.a;
realpart = real(input);
imagpart = imag(input);

output = [realpart; imagpart];

end
