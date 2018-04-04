% repeat matrix in similar way as repmat() in Matlab. 
%
classdef PermuteNode < GraphNode
    properties
        dimOrder = [1 2 3 4];         % order of dimensions in the new tensor
        reverseDimOrder = [1 2 3 4];
    end
    methods
        function obj = PermuteNode(dimOut, dimOrder)
            obj = obj@GraphNode('Permute',dimOut);
            obj.dimOrder = dimOrder;
            [~, obj.reverseDimOrder] = sort(dimOrder);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            obj.a = permute(input,obj.dimOrder);         
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            future_grad = obj.GetFutureGrad(future_layers);
            obj.grad{1} = permute(future_grad, obj.reverseDimOrder);
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end