
% create a sub network that takes in waveforms and produces fourier
% coefficients

function layer = genNetworkSTFT2LogSpec(stftLayer, useLog, logConst)
layer = stftLayer;

layer{end+1} = PowerNode(stftLayer{end}.dim(1));
if useLog
    if nargin<3
        logConst = 0.00;
    end
    layer{end+1} = LogarithmNode(layer{end}.dim(1), logConst);
end

layer = ConnectLinearGraph(layer);
layer = FinishLayer_obj(layer);

end
