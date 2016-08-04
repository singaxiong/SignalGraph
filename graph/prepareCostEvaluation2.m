% This function is intended to be called from mean square erorr or cross
% entropy cost function layers. It decides which input stream is system 
% output and which is desired target and scale of frames.
% 
% The function also applies following operations on the output and target: 
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
    [mask, variableLength] = getValidFrameMask(input_layers{1});     % check whether we have multiple segments of different lengths
else variableLength=0; mask = []; 
end

if variableLength   % if we have multiple sequences in the minibatch and they have different lengths
    [D,nFr,nSeg] = size(output);
    nFrOrig = nFr;
    [output] = ExtractVariableLengthTrajectory(output, mask);
    [target] = ExtractVariableLengthTrajectory(target, mask);
    if hasScale; [scale] = ExtractVariableLengthTrajectory(scale, mask); end
    for i=1:nSeg
        if size(target{i},2)==1
            if size(output{i},2)>1
                target{i} = repmat(target{i}, 1, size(output{i},2));
            end
        end
    end
    output = cell2mat_gpu(output);
    target = cell2mat_gpu(target);
    if hasScale; scale = cell2mat_gpu(scale); end
else
    [D,nFr,nSeg] = size(output);
    nFrOrig = nFr;
    if nFr>1 && size(target,2)==1    % if the frames share one target, replicate the target
        target = repmat(target, [1 nFr 1]);
    elseif size(target,2)>nFr
        target = target(:,1:nFr,:);
    end
    
    if nSeg>1     % if we are having multiple sequences in one minibatch, reshape it to an 2D matrix for easier downstream processing
        output = reshape(output, D, size(output,2)*nSeg);
        target = reshape(target, size(target,1),size(target,2)*nSeg);
        if hasScale; scale = reshape(scale, size(scale,1),size(scale,2)*nSeg); end
    end
end

end