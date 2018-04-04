% repeat matrix in selected dimension by specified times
%
classdef RepmatNode < GraphNode
    properties
        dimIdx = 1;         % which dimension to apply repmat
        repeatTimes = 2;     % how many times do we want to repeat
    end
    methods
        function obj = RepmatNode(dimOut, dimIdx, repeatTimes)
            obj = obj@GraphNode('Repmat',dimOut);
            obj.repeatTimes = repeatTimes;
            obj.dimIdx = dimIdx;
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            r = ones(1,4);
            r(obj.dimIdx) = obj.repeatTimes;
            obj.a = repmat(input, r(1), r(2), r(3), r(4));
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            future_grad = GetFutureGrad(future_layers, obj);
            [D(1),D(2),D(3),D(4)] = size(input);
            D2 = [D(1:obj.dimIdx) obj.repeatTimes D(obj.dimIdx:end)];
            future_grad = reshape(future_grad, D2);
            obj.grad{1} = squeeze(sum(future_grad, obj.dimIdx+1));
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
    end
    
end