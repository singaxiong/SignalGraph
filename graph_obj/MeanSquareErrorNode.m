classdef MeanSquareErrorNode < GraphNode
    
    methods
        function obj = MeanSquareErrorNode(myIdx)
            obj = obj@GraphNode('MeanSquareError', myIdx);
        end
        
        function obj = forward(obj,prev_layers)
            output = prev_layers{1}.a;
            target = prev_layers{2}.a;
            diff = output - target;
            obj.a = 0.5/m * sum(sum( diff .* conj(diff) ));   % support both real and complex numbers
            obj = forward@GraphNode(prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            output = prev_layers{1}.a;
            target = prev_layers{2}.a;
            diff = output - target;
            obj.grad = diff / m;
        end
        
    end
    
end