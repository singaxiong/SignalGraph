classdef SpliceNode < GraphNode
    properties
        context = 0;
    end
    
    methods
        function obj = SpliceNode(dimIn, context)
            dimOut = length(context)*dimIn;
            obj = obj@GraphNode('Splice',dimOut);
            obj.dim(2) = dimIn;
            obj.context = context;
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            
            input = prev_layers{1}.a;
            [D,T,N] = size(input);
            
            if length(obj.context) == 1     % if only 1 context, output the current frame
                obj.a = input;
            elseif N==1
                obj.a = ExpandContext_v2(input, obj.context);
            else
                if obj.variableLength
                    input = obj.PadShortTrajectory(input, 'last');
                end
                obj.a = ExpandContext_v2(input, obj.context);
            end
            
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            future_grad = obj.GetFutureGrad(future_layers);
            
            [D,T,N] = size(future_grad);
            
            if length(obj.context)==1
                obj.grad = future_grad;
                return;
            end
            
            if obj.variableLength
                future_grad = obj.PadShortTrajectory(future_grad, 0);
            end
            
            obj.grad = obj.AllocateMemoryLike([dim,T,N], future_grad);
            if obj.variableLength
                future_grad2 = obj.ExtractVariableLengthTrajectory(future_grad);
                for i=1:N
                    nFrSeg = size(future_grad2{i},2);
                    obj.grad(:,1:nFrSeg,i) = B_splice_single_sentence(obj.grad(:,1:nFrSeg,i), future_grad(:,1:nFrSeg,i), obj.context);
                end
            else
                obj.grad = B_splice_single_sentence(obj.grad, future_grad, obj.context);
            end
            
            obj.grad{1} = 1./(input+obj.const).*future_grad;
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
    end
    
    methods (Access = protected)
        function grad = B_splice_single_sentence(obj, grad, future_grad)
            [dim,nFr,nSeg] = size(future_grad);
            dim = dim/obj.context;
            half_ctx = (obj.context-1)/2;
            
            tmp1 = []; tmp2 = [];
            for i=-half_ctx:half_ctx
                curr_future_grad = future_grad( (i+half_ctx)*dim+1 : (i+half_ctx+1)*dim, :,:);
                if i<0
                    grad(:,1:end+i,:) = grad(:,1:end+i,:) + curr_future_grad(:, -i+1:end,:);
                    tmp1 = [tmp1 curr_future_grad(:, 1:-i,:)];
                elseif i>0
                    grad(:,i+1:end,:) = grad(:,i+1:end,:) + curr_future_grad(:, 1:end-i,:);
                    tmp2 = [tmp2 curr_future_grad(:, end-i+1:end,:)];
                else
                    grad = grad + curr_future_grad;
                end
            end
            grad(:,1,:) = grad(:,1,:) + sum(tmp1,2);
            grad(:,end,:) = grad(:,end,:) + sum(tmp2,2);
            
        end
    end
end