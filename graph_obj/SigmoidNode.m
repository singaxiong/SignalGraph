classdef SigmoidNode < GraphNode
    
    methods
        function obj = SigmoidNode(myIdx)
            obj = obj@GraphNode('Sigmoid', myIdx);
        end
        
        function obj = forward(obj,prev_layers)
            input = prev_layers{1}.a;
            obj.a = sigmoid(input);
            obj = forward@GraphNode(prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            if obj.skipGrad || obj.skipBP
                return;
            end
            
            future_grad = GetFutureGrad(future_layers);
            
            if isfield(curr_layer, 'rho')
                tmp = -obj.L1target./max(1e-3,obj.rho) + (1-obj.L1target)./max(1e-3,(1-obj.rho));
                future_grad = future_grad + repmat(obj.L1weight * tmp, 1, size(future_grad,2));
            end
            
            obj.grad = future_grad .* obj.a .* (1-obj.a);
        end
        
    end
    
end