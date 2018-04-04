classdef SigmoidNode < GraphNode
    
    methods
        function obj = SigmoidNode(dimOut)
            obj = obj@GraphNode('Sigmoid',dimOut);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            
            input = prev_layers{1}.a;
            obj.a = sigmoid(input);
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            if obj.skipGrad || obj.skipBP
                return;
            end
            
            future_grad = obj.GetFutureGrad(future_layers);
            
            if obj.L1weight>0 
                tmp = -obj.L1target./max(1e-3,obj.rho) + (1-obj.L1target)./max(1e-3,(1-obj.rho));
                future_grad = future_grad + repmat(obj.L1weight * tmp, 1, size(future_grad,2));
            end
            
            obj.grad{1} = future_grad .* obj.a .* (1-obj.a);

            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end