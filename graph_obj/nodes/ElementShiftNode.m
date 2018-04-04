% shift elements from specified dimension
% e.g. if dimIdx = 2, shift = 1, the class shifts the
% elements in dimension 2 by 1 to the right. 
%
classdef ElementShiftNode < GraphNode
    properties
        dimIdx = 1;         % which dimension to select from
        shift;   % which elements to select
        circular = 0;       % whether to use circular shift
        shiftIdx;
    end
    methods
        function obj = ElementShiftNode(shift, dimIdx, circular)
            obj = obj@GraphNode('ElementShift',length(shift));
            obj.shift = shift;
            if nargin>=2
                obj.dimIdx = dimIdx;
            end
            if nargin>=3
                obj.circular = circular;
            end
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            [D(1), D(2), D(3), D(4)] = size(input);
            nElement = D(obj.dimIdx);
            assert(nElement >= obj.shift, ...
                sprintf('%s:forward, Error: shift (%d) is larger than data length (%d), exit. \n', obj.name, obj.shift, nElement));
            obj.shiftIdx = 1:nElement;
            obj.shiftIdx = obj.shiftIdx + obj.shift;
            if obj.circular
                tmp = obj.shiftIdx>nElement;
                obj.shiftIdx(tmp) = obj.shiftIdx(tmp) - nElement;
                tmp = obj.shiftIdx<1;
                obj.shiftIdx(tmp) = obj.shiftIdx(tmp) + nElement;
            else
                obj.shiftIdx(obj.shiftIdx>nElement) = [];
                obj.shiftIdx(obj.shiftIdx<1) = [];
            end
            
            if obj.dimIdx==1
                obj.a = input(obj.shiftIdx,:,:,:);
            elseif obj.dimIdx==2
                obj.a = input(:,obj.shiftIdx,:,:);
            elseif obj.dimIdx==3
                obj.a = input(:,:,obj.shiftIdx,:);
            elseif obj.dimIdx==4
                obj.a = input(:,:,:,obj.shiftIdx);
            end                
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            future_grad = GetFutureGrad(future_layers, obj);
            
            input = prev_layers{1}.a;
            [D1,D2,T,N] = size(input);
            
            obj.grad{1} = obj.AllocateMemoryLike([D1 D2 T N], future_grad);
            if obj.dimIdx==1
                obj.grad{1}(obj.shiftIdx,:,:,:) = future_grad;
            elseif obj.dimIdx==2
                obj.grad{1}(:,obj.shiftIdx,:,:) = future_grad;
            elseif obj.dimIdx==3
                obj.grad{1}(:,:,obj.shiftIdx,:) = future_grad;
            elseif obj.dimIdx==4
                obj.grad{1}(:,:,:,obj.shiftIdx) = future_grad;
            end
        end
        
    end
    
end