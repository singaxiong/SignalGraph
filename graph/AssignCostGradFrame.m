function [grad] = AssignCostGradFrame(grad, nFrOrig, nSeg, mask, Cost_layer)
D = size(grad,1);

words = ExtractWordsFromString_v2(Cost_layer.costFrameSelection);
selectionType = words{1};

switch selectionType
    case 'last'
        if length(words)>1
            N = str2num(words{2});
        else
            N = 1;
        end
        
        gradTmp = grad;
        precision = class(gather(grad(1,1,1)));
        if strcmpi(class(grad), 'gpuArray')
            grad = gpuArray.zeros(D,nFrOrig, nSeg, precision);
        else
            grad = zeros(D,nFrOrig, nSeg, precision);
        end
        if numel(mask)>0 && sum(sum(mask))>0   % if the trajectories have variable length
            for i=1:nSeg
                idx = find(mask(:,i)==1);
                if isempty(idx)
                    grad(:,max(1,end-N+1):end,i) = gradTmp(:,:,i);
                else
                    grad(:,max(1,idx(1)-N):max(1,idx(1)-1),i) = gradTmp(:,:,i);
                end
            end
            grad = PadShortTrajectory(grad, mask, -1e10);
        else
            grad(:,max(1,end-N+1):end,:) = gradTmp;
        end
end
end