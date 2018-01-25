classdef InputNode < GraphNode
    properties
    end
    methods
        function obj = InputNode(inputIndex, dimOut)
            obj = obj@GraphNode('input', dimOut);
            obj.prev = inputIndex;
            obj.dim(2) = dimOut;
        end
    end
    
end