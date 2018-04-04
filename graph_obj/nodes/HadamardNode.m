% elementwise production of all previous layers, which are required to have the
% same size. 
% 
classdef HadamardNode < GraphNode
    
    methods
        function obj = HadamardNode(dimOut)
            obj = obj@GraphNode('Hadamard',dimOut);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            
            obj.a = prev_layers{1}.a;
            for i=2:length(prev_layers)
                obj.a = obj.a .* conj(prev_layers{i}.a);
            end 
            
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            if obj.skipGrad || obj.skipBP
                return;
            end
            
            future_grad = obj.GetFutureGrad(future_layers);
            for i=1:length(prev_layers)
                [D(1) D(2) D(3) D(4)] = size(prev_layers{i}.a);
                if i==1
                    grad = future_grad .* prev_layers{2}.a;
                else
                    grad = future_grad .* prev_layers{1}.a;
                end
                for j = 1:4
                    if D(j) == 1
                        grad = sum(grad,j);
                    end
                end
                obj.grad{i} = conj(grad);
            end

            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end