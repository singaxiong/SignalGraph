% multiply -1 to the activation
% 
classdef NegativeNode < GraphNode
    
    methods
        function obj = NegativeNode(dimOut)
            obj = obj@GraphNode('Negative',dimOut);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            
            obj.a = -prev_layers{1}.a;
            
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            if obj.skipGrad || obj.skipBP
                return;
            end
            
            future_grad = obj.GetFutureGrad(future_layers);
            obj.grad{1} = -future_grad;

            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end