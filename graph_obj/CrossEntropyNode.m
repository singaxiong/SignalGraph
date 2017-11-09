classdef CrossEntropyNode < GraphNode
    properties
        acc = [];   % accuracy of classification
    end
    methods
        function obj = CrossEntropyNode(myIdx)
            obj = obj@GraphNode('CrossEntropy', myIdx);
        end
        
        function obj = forward(obj,prev_layers)
            output = prev_layers{1}.a;
            trueClass = prev_layers{2}.a;

            m = size(output,2);
            [~, recogClass] = max(output);
            acc2 = recogClass==trueClass;
            obj.acc = sum(acc2)/m;
            
            dim = size(output,1);
            offset = 0:dim:m*dim-1;
            idx = offset+double(trueClass);     % critical: we need to use double for trueClass as single has limited precision and will fail for very larger numbers
            obj.a = output(idx);
            cost = -1/m*sum(log(obj.a));
            
            % check if cost is nan. If any element in output2 is 0, cost will be
            % nan. We need to prevent this to make the training continue.
            if isnan(cost)
                obj.a = max(obj.a, eps);
            end
            
            obj = forward@GraphNode(prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            output = prev_layers{1}.a;
            target = prev_layers{2}.a;
            m = size(output,2);
            obj.grad = -1/m * target ./ output;
        end
        
    end
    
end