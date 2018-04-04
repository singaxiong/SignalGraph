classdef MuLawNode < GraphNode
    properties
        mu = 255;
    end
    
    methods
        function obj = MuLawNode(dimOut)
            obj = obj@GraphNode('MuLaw',dimOut);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            
            obj.a = sign(input) .* log(1+obj.mu*abs(input)) ./ log(1+obj.mu);

            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            if obj.skipGrad || obj.skipBP
                return;
            end
            
            future_grad = obj.GetFutureGrad(future_layers);
            % to be implemented

            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end