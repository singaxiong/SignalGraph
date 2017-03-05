function output = F_ConcatRealImag(prev_layer)

covMat = prev_layer.a;

output = [real(covMat); imag(covMat)];

end
