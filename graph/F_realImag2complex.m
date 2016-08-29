% assume the input is a DxTxN tensor. The first D/2 dimensions are store the
% real parts, and the last D/2 dimensions store the imaginary parts. The
% function returns the corresponding complex numbers
%
function [output] = F_realImag2complex(input_layer)
input = input_layer.a;
[D, T, N] = size(input);
j = sqrt(-1);

realpart = input(1:D/2,:,:);
imagpart = input(D/2+1:end,:,:);

output = realpart + j*imagpart;
end
