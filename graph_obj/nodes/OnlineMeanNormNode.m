classdef OnlineMeanNormNode < GraphNodeUpdatable
    properties
        alpha = 0.9;
        x0 = [];     % initial value of input average
        W = [];
        b = [];
    end
    
    methods
        function obj = OnlineMeanNormNode(dimOut)
            obj = obj@GraphNodeUpdatable('OnlineMeanNorm',dimOut);
        end
        
        function obj = initialize(obj, alpha)
            obj.x0 = ones(obj.dim(1),1)*0;
            if nargin>=2
                obj.W = ones(obj.dim(1),1) * alpha;
            else
                obj.W = ones(obj.dim(1),1) * obj.alpha;
            end
            obj.b = ones(obj.dim(1),1)*0;
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            
            input = prev_layers{1}.a;
            [D,T,N] = size(input);
            
            xbar = obj.AllocateMemoryLike([D N T], input);
            xbar(:,:,1) = repmat(obj.x0, 1,N);
            
            W2 = 1-obj.W;
            for t=2:T
                xbar(:,:,t) = xbar(:,:,t-1) * obj.W + input2(:,:,t) .* W2;
            end
            
            obj.a = input - permute(xbar, [1 3 2]);
            
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            future_grad = obj.GetFutureGrad(future_layers);
            
            [D,T,N] = size(future_grad);
            
            if N==1
            else
            end
            
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
    end
end