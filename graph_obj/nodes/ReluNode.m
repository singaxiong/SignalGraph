classdef ReluNode < GraphNode
    properties
        threshold = 0;
    end
    
    methods
        function obj = ReluNode(dimOut, threshold)
            obj = obj@GraphNode('Relu',dimOut);
            if nargin>=2
                obj.threshold = threshold;
            end
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            obj.a = max(obj.threshold, input);
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            if obj.skipGrad || obj.skipBP
                return;
            end
            
            future_grad = obj.GetFutureGrad(future_layers);
            output = obj.a;

            mask = output>obj.threshold;
            obj.grad{1} = future_grad;
            obj.grad{1}(mask==0) = 0;

            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end