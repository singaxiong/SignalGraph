function [grad] = AssignCostGradFrame(grad, nFrOrig, nSeg, mask, Cost_layer)
D = size(grad,1);

switch Cost_layer.costFrameSelection
    case 'last'
        gradTmp = grad;
        precision = class(gather(grad(1,1,1)));
        if strcmpi(class(grad), 'gpuArray')
            grad = gpuArray.zeros(D,nFrOrig, nSeg, precision);
        else
            grad = zeros(D,nFrOrig, nSeg, precision);
        end
        if numel(mask)>0    % if the trajectories have variable length
            for i=1:nSeg
                idx = find(mask(:,i)==1);
                if isempty(idx)
                    grad(:,end,i) = gradTmp(:,1,i);
                else
                    grad(:,idx(1)-1,i) = gradTmp(:,1,i);
                end
            end
            grad = PadShortTrajectory(grad, mask, -1e10);
        else
            grad(:,end,:) = gradTmp;
        end
end
end