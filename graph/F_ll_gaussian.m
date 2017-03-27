% take the covariance matrix of input trajectories

function [output, validFrameMask] = F_ll_gaussian(prev_layers, curr_layer)

mu = prev_layers{1}.a;
variance = prev_layers{2}.a;
input = prev_layers{3}.a;

LogLikelihood = -(input-mu).^2 ./ variance /2; 

LogLikelihood = LogLikelihood - 0.5 * log(2*variance*pi);

LogLikelihood = squeeze(sum(LogLikelihood));

[D,T,N] = size(input);

if N==1
    AvgLogLikelihood = mean(LogLikelihood);
    validFrameMask = [];
else
    [validFrameMask, variableLength] = getValidFrameMask(prev_layers{3});
    mask = 1-validFrameMask;
    AvgLogLikelihood = LogLikelihood .* mask;
    AvgLogLikelihood = sum(AvgLogLikelihood(:)) / sum(mask(:));
end

% we will use the negative log likelihood as the cost function
output = -AvgLogLikelihood;

end
