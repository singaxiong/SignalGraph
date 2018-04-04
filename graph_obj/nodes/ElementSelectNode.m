% select elements from specified dimension
% e.g. if dimIdx = 2, elementSelectIdx = [1 3], the class selects the
% elements 1 and 3 from dimension 2 of the input tensor. 
%
classdef ElementSelectNode < GraphNode
    properties
        dimIdx = 1;         % which dimension to select from
        elementSelectIdx;   % which elements to select
    end
    methods
        function obj = ElementSelectNode(elementSelectIdx, dimIdx)
            obj = obj@GraphNode('ElementSelect',length(elementSelectIdx));
            obj.elementSelectIdx = elementSelectIdx;
            if nargin>=2
                obj.dimIdx = dimIdx;
            end
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            if obj.dimIdx==1
                obj.a = input(obj.elementSelectIdx,:,:,:);
            elseif obj.dimIdx==2
                obj.a = input(:,obj.elementSelectIdx,:,:);
            elseif obj.dimIdx==3
                obj.a = input(:,:,obj.elementSelectIdx,:);
            elseif obj.dimIdx==4
                obj.a = input(:,:,:,obj.elementSelectIdx);
            end                
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            future_grad = GetFutureGrad(future_layers, obj);
            
            input = prev_layers{1}.a;
            [D1,D2,T,N] = size(input);
            
            obj.grad{1} = obj.AllocateMemoryLike([D1 D2 T N], future_grad);
            if obj.dimIdx==1
                obj.grad{1}(obj.elementSelectIdx,:,:,:) = future_grad;
            elseif obj.dimIdx==2
                obj.grad{1}(:,obj.elementSelectIdx,:,:) = future_grad;
            elseif obj.dimIdx==3
                obj.grad{1}(:,:,obj.elementSelectIdx,:) = future_grad;
            elseif obj.dimIdx==4
                obj.grad{1}(:,:,:,obj.elementSelectIdx) = future_grad;
            end
        end
        
    end
    
end