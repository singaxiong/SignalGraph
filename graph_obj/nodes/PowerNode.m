classdef PowerNode < GraphNode
    
    methods
        function obj = PowerNode(dimOut)
            obj = obj@GraphNode('Power',dimOut);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            
            input = prev_layers{1}.a;
            obj.a = abs( input .* conj(input) );
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj, prev_layers, future_layers)
            future_grad = obj.GetFutureGrad(future_layers);
            input = prev_layers{1}.a;
            obj.grad{1} = 2 * future_grad .* conj(input);      % see eq (211) of matrix cookbook 2008 version.
            obj.grad{1} = conj(obj.grad{1});
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end