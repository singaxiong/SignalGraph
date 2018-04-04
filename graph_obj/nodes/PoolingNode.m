classdef PoolingNode < GraphNode
    properties
        poolingDimension = 3;      % poolinig happens along this dimension
        poolingType = 'max';       % [max|min|mean|median|sum]
        selectedIdx = [];          % remember the index of data selected by pooling in [max|min|median]
    end
    
    methods
        function obj = PoolingNode(dimOut, poolingType, poolingDimension)
            obj = obj@GraphNode('Pooling',dimOut);
            if nargin>=2
                obj.poolingType = poolingType;
            end
            if nargin>=3
                obj.poolingDimension = poolingDimension;
            end
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            
            switch lower(obj.poolingType)
                case 'max'
                    [obj.a, obj.selectedIdx] = max(input, [], obj.poolingDimension);
                case 'min'
                    [obj.a, obj.selectedIdx] = min(input, [], obj.poolingDimension);
                case 'mean'
                    obj.a = mean(input, obj.poolingDimension);
                case 'median'
                    obj.a = median(input, obj.poolingDimension);
                    % need to implement selectedIdx by myself in the future
                case 'sum'
                    obj.a = sum(input, obj.poolingDimension);
            end
            
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            if obj.skipGrad || obj.skipBP
                return;
            end
            
            future_grad = obj.GetFutureGrad(future_layers);
            input = prev_layers{1}.a;
            [D(1), D(2), D(3), D(4)] = size(input);
            
            switch lower(obj.poolingType)
                case {'max', 'min', 'median'}
                    obj.grad{1} = obj.AllocateMemoryLike(D, future_grad);
                    obj.grad{1}(obj.selectedIdx) = future_grad;
                case {'mean','sum'}
                    if strcmpi(obj.poolingType, 'mean')
                        future_grad = future_grad / D(obj.poolingDimension);
                    end
                    shape = ones(1,4);
                    shape(obj.poolingDimension) = D(obj.poolingDimension);
                    obj.grad{1} = repmat(future_grad, shape);
            end
            
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end