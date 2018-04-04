classdef CrossEntropyNode < GraphNodeCost
    properties
        acc = [];   % accuracy of classification
    end
    methods
        function obj = CrossEntropyNode(costWeight)
            obj = obj@GraphNodeCost('CrossEntropy', costWeight);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);

            output = prev_layers{1}.a;
            [D(1) D(2) D(3) D(4)] = size(output);
            trueClass = prev_layers{2}.a;
            [N(1) N(2) N(3) N(4)] = size(trueClass);
            if D(3)>1 && N(3)==1
                trueClass = repmat(trueClass, 1, 1, D(3), 1);
            end
            trueClass = trueClass(:,:);
            output = output(:,:);
            
            m = size(output,2);
            [~, recogClass] = max(output);
            acc2 = recogClass==trueClass;
            obj.acc = sum(acc2)/m;
            
            dim = size(output,1);
            offset = 0:dim:m*dim-1;
            idx = offset+double(trueClass);     % critical: we need to use double for trueClass as single has limited precision and will fail for very larger numbers
            obj.a = output(idx);
            obj.a = -1/m*sum(log(obj.a));
            
            % check if cost is nan. If any element in output2 is 0, cost will be
            % nan. We need to prevent this to make the training continue.
            if isnan(obj.a)
                obj.a = max(obj.a, eps);
            end
            
            obj = forward@GraphNodeCost(obj, prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            output = prev_layers{1}.a;
            target = prev_layers{2}.a;
            m = size(output,2);
            obj.grad{1} = -1/m * target ./ output;
            obj = backward@GraphNodeCost(obj, prev_layers, future_layers);
        end
        
    end
    
end