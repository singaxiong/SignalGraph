% repeat matrix in similar way as repmat() in Matlab. 
%
classdef ReshapeNode < GraphNode
    properties
        sourceDimSizes;
        targetDimSizes;
    end
    methods
        function obj = ReshapeNode(dimOut, sourceDimSizes, targetDimSizes)
            obj = obj@GraphNode('Reshape',dimOut);
            obj.sourceDimSizes = sourceDimSizes;
            obj.targetDimSizes = targetDimSizes;
            assert(prod(obj.sourceDimSizes)==prod(obj.targetDimSizes), ...
                sprintf('%s:%sNode, Error: source dimension sizes [%s] not equal to target dimension sizes [%s]', ...
                obj.name, obj.name, num2str(obj.sourceDimSizes), num2str(obj.targetDimSizes)));
            obj.dim(1:2) = targetDimSizes(1:2);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            [D(1), D(2), D(3), D(4)] = size(input);
            reshapeSize = [obj.targetDimSizes(:)' D(length(obj.sourceDimSizes)+1:end)];
            obj.a= reshape(input, reshapeSize);
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            future_grad = obj.GetFutureGrad(future_layers);
            input = prev_layers{1}.a;
            [D1,D2,T,N] = size(input);
            obj.grad{1} = reshape(future_grad, D1,D2,T,N);
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end