classdef InputNode < GraphNode
    methods
        function obj = InputNode(myIdx, inputIndex)
            obj = obj@GraphNode('input', myIdx);
            obj.prev = inputIndex;
        end
    end
    
end