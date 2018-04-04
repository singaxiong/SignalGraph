% concatenate the output of previous layers in specified dimension
%
classdef ConcatNode < GraphNode
    properties
        dimCat = 1;     % along which dimension the concatenation is applied
    end
    methods
        function obj = ConcatNode(dimOut, dimCat)
            obj = obj@GraphNode('Concat',dimOut);
            obj.dimCat = dimCat;
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            obj.a = prev_layers{1}.a;
            for i=2:length(prev_layers)
                obj.a = cat(obj.dimCat, obj.a, prev_layers{i}.a);
            end
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            future_grad = GetFutureGrad(future_layers, obj);
            
            for i=1:length(prev_layers)
                [D(1,i), D(2,i), D(3,i), D(4,i)] = size(prev_layers{i}.a);
                idx = 1+ sum( D(obj.dimCat,1:i-1) ) : sum( D(obj.dimCat,1:i) );
                if obj.dimCat == 1
                    obj.grad{i} = future_grad(idx, :,:,:);
                elseif obj.dimCat == 2
                    obj.grad{i} = future_grad(:,idx,:,:);
                elseif obj.dimCat == 3
                    obj.grad{i} = future_grad(:,:,idx,:);
                elseif obj.dimCat == 4
                    obj.grad{i} = future_grad(:,:,:,idx);
                end
            end
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end