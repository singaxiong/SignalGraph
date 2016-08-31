function output = F_SpatialCov(input_layer, curr_layer)

input = input_layer.a;
[D,T,N] = size(input);

curr_layer = SetDefaultValue(curr_layer, 'winSize', 0);
curr_layer = SetDefaultValue(curr_layer, 'winShift', 1);

nBin = length(curr_layer.freqBin);
nCh = D/nBin;

if N==1
    input2 = reshape(input, nBin, nCh, T,N);
    R = ComplexSpectrum2SpatialCov(input2, curr_layer.winSize, curr_layer.winShift);
    output = permute(R, [3 1 2 4]);
    output = reshape(output, nBin*nCh^2, size(output,4),N);
else
    % to be implemented
end

end