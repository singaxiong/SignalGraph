
% create a sub network that takes in waveforms and produces fourier
% coefficients

function layer = genNetworkSTFT(input_idx, nCh, usedChannel, nFFT)
layer{1} = InputNode(input_idx, nCh);
nUsedChannel = length(usedChannel);
if nUsedChannel < nCh
    layer{end+1} = ElementSelectNode(usedChannel);
end

featDim = nUsedChannel*(nFFT/2+1);
layer{end+1} = STFTNode(featDim);
layer{end+1} = AffineNode(featDim);
layer{end}.W = eye(featDim)*1;
layer{end}.b = zeros(featDim,1);
layer{end} = layer{end}.setUpdate(0,0);
layer = ConnectLinearGraph(layer);
layer = FinishLayer_obj(layer);

end
