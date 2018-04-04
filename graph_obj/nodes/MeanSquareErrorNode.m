classdef MeanSquareErrorNode < GraphNodeCost
    properties
        
    end
    
    methods
        function obj = MeanSquareErrorNode(weight)
            obj = obj@GraphNodeCost('MeanSquareError', weight);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            
            output = prev_layers{1}.a;
            target = prev_layers{2}.a;
            diff = output - target;
            [D,T,N] = size(output);
            obj.a = real(0.5/T/N * sum( diff(:) .* conj(diff(:)) ));   % support both real and complex numbers
            obj = forward@GraphNodeCost(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            input1 = prev_layers{1}.a;
            input2 = prev_layers{2}.a;
            diff = input1 - input2;
            [D,T,N] = size(input1);
            obj.grad{1} = diff / (T*N);
            obj.grad{2} = -obj.grad{1};
            obj = backward@GraphNodeCost(obj, prev_layers, future_layers);
        end
        
    end
    
end