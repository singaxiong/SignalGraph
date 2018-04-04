% take the exponetiation of elements of input tensor
% e.g. if order = -0.5, output = input .^ (-0.5) = 1 ./ sqrt(input). 
%
classdef ExponentiationNode < GraphNode
    properties
        order = -1;
    end
    methods
        function obj = ExponentiationNode(dimOut, order)
            obj = obj@GraphNode('Exponentiation',dimOut);
            if nargin>=2
                obj.order = order;
            end
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            
            input = prev_layers{1}.a;
            obj.a = input .^ obj.order;
            
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj, prev_layers, future_layers)
            future_grad = obj.GetFutureGrad(future_layers);
            input = prev_layers{1}.a;
            
            obj.grad{1} = obj.order * input .^ obj.order .* future_grad;
            
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end