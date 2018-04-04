classdef InputNode < GraphNode
    properties
    end
    methods
        function obj = InputNode(inputIndex, dimOut)
            obj = obj@GraphNode('Input', dimOut);
            obj.prev = inputIndex;
            obj.dim(3:4) = obj.dim(1:2);
        end
    end
    
end