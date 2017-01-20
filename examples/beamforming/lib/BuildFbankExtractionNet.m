% Build and initialize the computational graph for extracting log Mel
% filterbank features
%
function [layer, para] = BuildFbankExtractionNet()
para.output = 'tmp';
para.IO.nStream = 1;
para.NET.sequential = 1;
para.cost_func.layer_idx = [];

para = ConfigBasicSTFT(para);
layer = genNetworkFbankExtraction(para.topology);     % generate the network graph

% generating the scaling factor for the input, as we will need to use a
% small constant in the logarithm. We need to make sure that the power of
% speech are larger than this constant most of the time. 
scale = 1e4;        % we hard code the scale to be a constant so that all network will use the same number
scale = scale/2^16; % note that we are using int16 to store waveform samples, so need to scale down
layer = InitWavScaleLayer(layer, scale);

% set Mel filterbank linear transform
layer = InitMelLayer(layer, para);

para.out_layer_idx = length(layer);

para = ParseOptions2(para);

end
