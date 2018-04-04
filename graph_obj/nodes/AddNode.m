% add activation of all previous layers, which are required to have the
% same size. 
% 
classdef AddNode < GraphNode
    
    methods
        function obj = AddNode(dimOut)
            obj = obj@GraphNode('Add',dimOut);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            obj.a = prev_layers{1}.a;
            for i=2:length(prev_layers)
                obj.a = obj.a + prev_layers{i}.a;
            end
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            if obj.skipGrad || obj.skipBP
                return;
            end
            
            future_grad = obj.GetFutureGrad(future_layers);
            for i=1:length(prev_layers)
                obj.grad{i} = future_grad;
            end

            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end