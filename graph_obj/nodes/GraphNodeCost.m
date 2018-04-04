% Class for cost nodes. 
% Cost nodes all have one or more input nodes, but not followed by other
% nodes. Every cost nodes should have a weight, which is the weight of the
% cost node in the overall cost function of the network. 
classdef GraphNodeCost < GraphNode
    properties
        costWeight = 1;     % weight of the cost function
    end
    
    methods
        function obj = GraphNodeCost(name, costWeight)
            obj = obj@GraphNode(name, 1);
            obj.costWeight = costWeight;
        end
        function obj = forward(obj, prev_layers)
            obj.a = obj.a * obj.costWeight;
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj, prev_layers, future_layers)
            if obj.costWeight ~= 1
                for i=1:length(obj.grad)
                    obj.grad{i} = obj.grad{i} * obj.weight;
                end
            end
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
    end
    
    methods (Access = protected)
    end
end