% the activation is the same as the weight. Usually used to introduce free
% parameters to the graph from nowhere. 
% 
classdef Weight2ActivationNode < GraphNodeUpdatable
    properties
    end
    methods
        function obj = Weight2ActivationNode(dimOut, dimIn)
            obj = obj@GraphNodeUpdatable('Weight2Activation',dimOut);
            obj.dim(2) = dimIn;
            obj.prev = 0;   % there is no previous layer
            obj.updateBias = 0;
        end
        
        function obj = initialize(obj, gauss, negbias, r)
            if sum(abs(size(obj.W)-obj.dim([1 2])))>0
                if gauss
                    obj.W = 3/sqrt(obj.dim(2)) * randn(obj.dim([1 2]));
                else
                    obj.W = rand(obj.dim([1 2])) * 2 * r - r;
                end
            end            
        end
        
        function obj = forward(obj)
            obj.a = obj.W;
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            future_grad = obj.GetFutureGrad(future_layers);
            obj.gradW = future_grad;
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end