% compute log determinant of input matriecs D1 x D2. Assume D1==D2. 
% Used in maximum likelihood estimations. 
%
classdef LogDetNode < GraphNodeCost
    properties
    end
    methods
        function obj = LogDetNode(costWeight)
            obj = obj@GraphNodeCost('LogDet', costWeight);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);

            input = prev_layers{1}.a;
            
            precision = class(gather(input(1)));
            if ~strcmpi(precision, 'double')    % we need to use double precision
                input = double(input);
            end
            
            [D1,D2,M,N] = size(input);
            obj.a = obj.AllocateMemoryLike([M N], input);
            for i=1:M       % will change the implementation to cellfun for efficiency
                for j=1:N
                    obj.a(i,j) = log(det(input(:,:,i,j)));
                end
            end
            
            obj.a = reshape(obj.a, 1,1,M,N);
            if strcmpi(precision, 'single')
                obj.a = single(obj.a);
            end
            
            obj = forward@GraphNodeCost(prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            input = prev_layers{1}.a;
            
            precision = class(gather(input(1)));
            if ~strcmpi(precision, 'double')    % we need to use double precision
                input = double(input);
            end
            
            [D1,D2,M,N] = size(input);
            obj.grad{1} = obj.AllocateMemoryLike([M N], input);
            for i=1:M       % will change the implementation to cellfun for efficiency
                for j=1:N
                    obj.grad{1}(:,:,i,j) = inv(input(:,:,i,j));
                end
            end
            if strcmpi(precision, 'single')
                obj.grad{1} = single(obj.grad{1});
            end
            
            obj = backward@GraphNodeCost(obj, prev_layers, future_layers);
        end
        
    end
    
end