% This function is intended to be called from mean square erorr or cross
% entropy cost function layers. It decides which input stream is system 
% output and which is desired target and scale of frames.
% 
% The function also applies following operations on the output and target: 
%   1) labelDelay: delay the target for (negative) integer number of time
%   steps (frames).
%   2) costFrameSelection: only compute cost function for a selected number
%   of time steps (frames). Following ways to selecting frames is supported
%   or plan to be supported:
%       a) "last", use the last frame 
%       b) "last N", use the last N frames. 
%       c) "first" and "first N", first or first N frames. 
%       d) to be supported
%   3) scale: assign weights to different time steps (frames) in the cost
%   function. the "scale" matrix, if available, will be multiplied to the
%   time steps. In fact, "scale" can also achieve all the functionality of
%   costFrameSelection, but in sometimes, the later is easier to use. 
%   4) variableLength: in CNN or LSTM networks, a minibatch may consists of
%   segments that have different lengths. We need to take care of this in
%   this function. 
%
% Note that there is a function called PostprocessCostEvaluation() that
% reverses everything we have done in this function such that the gradient
% can be passed back in correct format. 
%
% Author: Xiong Xiao, Temasek Labs, NTU, Singapore. 
% Created: 2014
% Last Modified: 28 Jun 2016
%   
function [nSeg, output, target, scale, nFrOrig, mask] = prepareCostEvaluation(input_layers, Cost_layer)
% If there are two input layers, the second must be target, and the other is the system prediction.
% If there are 3 input layers, the third one will be the weights of frames.
output = input_layers{1}.a;
target = input_layers{2}.a;
scale = []; hasScale = 0;
if length(input_layers)==3
    scale = input_layers{3}.a;
    hasScale = 1;
end
[~,~,nSeg] = size(output);
if nSeg>1; 
    [mask, variableLength] = CheckTrajectoryLength(output);     % check whether we have multiple segments of different lengths
else variableLength=0; mask = []; 
end

if variableLength   % if we have multiple sequences in the minibatch and they have different lengths
    if isfield(Cost_layer, 'labelDelay') && Cost_layer.labelDelay~=0
        labelDelay = Cost_layer.labelDelay;   % labelDelay is the number of frames to shift.
        nFrActual = gather(sum(mask==0));   % we need to gather nFrActual as it will be used as index range and must not be in GPU memory
        % shift the target
        if labelDelay>0     % positive labelDelay means that we delay the prediction by labelDelay frames
            target = target(:,1:end-labelDelay, :);
            output = output(:,labelDelay+1:end,:);
            if hasScale; scale = scale(:,1:end-labelDelay,:); end
            for i=1:nSeg    % mark the end of sequences at correct places
                target(:,nFrActual(i)-labelDelay+1:end,i) = -1e10;
                if hasScale; scale(:,nFrActual(i)-labelDelay+1:end,i) = -1e10; end
            end
        else                % negative labelDelay means that we predict into the future by labelDelay frames
            target = target(:,labelDelay+1:end, :);
            output = output(:,1:end-labelDealy,:);
            if hasScale; scale = scale(:,labelDelay+1,:); end
            for i=1:nSeg
                output(1,nFrActual(i)-labelDelay+1:end,i) = -1e10;
            end
        end
    end
    
    [D,nFr,nSeg] = size(output);
    nFrOrig = nFr;
    [output] = ExtractVariableLengthTrajectory(output);
    [target] = ExtractVariableLengthTrajectory(target);
    if hasScale; [scale] = ExtractVariableLengthTrajectory(scale); end

    if isfield(Cost_layer, 'costFrameSelection')
        [output, target] = SelectCostFrame(output, target, Cost_layer);     % currently, we don't support costFrameSelection together with scale as this is not likely to happen
    end
    
    output = cell2mat_gpu(output);
    target = cell2mat_gpu(target);
    if hasScale; scale = cell2mat_gpu(scale); end
else
    if isfield(Cost_layer, 'labelDelay') && Cost_layer.labelDelay~=0
        labelDelay = Cost_layer.labelDelay;   % labelDelay is the number of frames to shift.
        % shift the target
        if labelDelay>0     % positive labelDelay means that we delay the prediction by labelDelay frames
            target = target(:,1:end-labelDelay, :);
            output = output(:,labelDelay+1:end,:);
            if hasScale; scale = scale(:,1:end-labelDelay,:); end
        else                % negative labelDelay means that we predict into the future by labelDelay frames
            target = target(:,labelDelay+1:end, :);
            output = output(:,1:end-labelDealy,:);
            if hasScale; scale = scale(:,labelDelay+1,:); end
        end
    end
    
    [D,nFr,nSeg] = size(output);
    nFrOrig = nFr;
    if nFr>1 && size(target,2)==1    % if the frames share one target, replicate the target
        target = repmat(target, [1 nFr 1]);
    elseif size(target,2)>nFr
        target = target(:,1:nFr,:);
    end
    
    if isfield(Cost_layer, 'costFrameSelection')
        [output, target] = SelectCostFrame(output, target, Cost_layer);     % currently, we don't support costFrameSelection together with scale as this is not likely to happen
    end
    
    if nSeg>1     % if we are having multiple sequences in one minibatch, reshape it to an 2D matrix for easier downstream processing
        output = reshape(output, D, size(output,2)*nSeg);
        target = reshape(target, size(target,1),size(target,2)*nSeg);
        if hasScale; scale = reshape(scale, size(scale,1),size(scale,2)*nSeg); end
    end
end

end