% This function decide which input is system output and which is desired target.
function [nSeg, output, target, scale, nFrOrig, mask] = prepareCostEvaluation(input_layers, Cost_layer)
% If there are two input layers, the second must be target, and the other is the system prediction.
% If there are 3 input layers, the third one will be the weights of frames.
output = input_layers{1}.a;
target = input_layers{2}.a;
scale = [];
if length(input_layers)==3
    scale = input_layers{3}.a;
end

if isfield(Cost_layer, 'labelDelay') && Cost_layer.labelDelay~=0
    labelDelay = Cost_layer.labelDelay;   % labelDelay is the number of frames to shift.
    % shift the target
    if labelDelay>0     % positive labelDelay means that we delay the prediction by labelDelay frames
        target = target(:,1:end-labelDelay, :);
        output = output(:,labelDelay+1:end,:);
    else                % negative labelDelay means that we predict into the future by labelDelay frames
        target = target(:,labelDelay+1:end, :);
        output = output(:,1:end-labelDealy,:);
    end
end

[D,nFr,nSeg] = size(output);
nFrOrig = nFr;

if nSeg>1; [mask, variableLength] = CheckTrajectoryLength(output); 
else variableLength=0; mask = []; end

if variableLength
    [output] = ExtractVariableLengthTrajectory(output);
    [target] = ExtractVariableLengthTrajectory(target);
else
    if nFr>1 && size(target,2)==1    % if the frames share one target, replicate the target
        target = repmat(target, [1 nFr 1]);
    elseif size(target,2)>nFr
        target = target(:,1:nFr,:);
    end
end    
if isfield(Cost_layer, 'costFrameSelection')
    [output, target] = SelectCostFrame(output, target, Cost_layer);
end

if nSeg>1     % if we are having multiple sequences in one minibatch, reshape it to an 2D matrix for easier downstream processing
    if variableLength
        output = cell2mat_gpu(output);
        target = cell2mat_gpu(target);
    else
        output = reshape(output, D, size(output,2)*nSeg);
        target = reshape(target, size(target,1),size(target,2)*nSeg);
    end
end
end