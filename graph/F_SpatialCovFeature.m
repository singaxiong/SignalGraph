function output = F_SpatialCovFeature(input_layers, curr_layer)

input = input_layers{1}.a;
[D,T,N] = size(input);

curr_layer = SetDefaultValue(curr_layer, 'winSize', 0);
curr_layer = SetDefaultValue(curr_layer, 'winShift', 1);

nBin = length(curr_layer.freqBin);
nCh = D/nBin;

if N==1
    output = ExtractSpatialCovFeat(input, nCh, curr_layer.winSize, curr_layer.winShift, curr_layer.binStep, curr_layer.SCMRowOnly, curr_layer.nChLogMag);
else
    for i=1:N
        feat = ExtractSpatialCovFeat(input(:,:,i), nCh, curr_layer.winSize, curr_layer.winShift, curr_layer.binStep, curr_layer.SCMRowOnly, curr_layer.nChLogMag);
        if i==1
            if strcmpi(class(input), 'gpuArray')
                output = gpuArray.zeros(size(feat,1), size(feat,2), N);
            else
                output = zeros(size(feat,1), size(feat,2), N);
            end
        end
        output(:,:,i) = feat;
    end
end

end