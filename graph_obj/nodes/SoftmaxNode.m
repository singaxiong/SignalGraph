classdef SoftmaxNode < GraphNode
    
    methods
        function obj = SoftmaxNode(dimOut)
            obj = obj@GraphNode('Softmax',dimOut);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            
            input = prev_layers{1}.a;
            % Sometimes, output's element may get to inf when single precision is used.
            % in such cases, we will have nan cross entropy. In such case, we can
            % simply ignore the minibatch in training.
            output = exp(input);
            recip_sum_a = 1./sum(output);
            obj.a = bsxfun(@times, output, recip_sum_a);

            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers, target)
            if obj.skipGrad || obj.skipBP
                return;
            end

            if length(future_layers)==1 && strcmpi(future_layers{1}.name, 'CrossEntropy')   % if softmax is followed by cross entropy node, compute their gradient together. 
                output = obj.a;
                [D(1),D(2),D(3),D(4)] = size(output);
                assert(D(2)==1, sprintf('%s:backward, Error: dimension 2 = %d not equal to 1', D(2)));
                output = output(:,:);
                m = D(3)*D(4);
                
                if size(target,3)==1 && D(3)>1  % for sequence, we may specify only one class for the entire sequence, e.g. in speaker ID. Need to repeat the class ID for all frames in the sequence. 
                    target = repmat(target, 1, 1, D(3), 1);
                end
                trueClass = reshape(target, 1, D(3)*D(4));
                offset = 0 : D(1) : m*D(1)-1;
                idx = offset+trueClass;
                
                output(idx) = output(idx) - 1;
                obj.grad{1} = reshape(output/m, D);
            else
                future_grad = obj.GetFutureGrad(future_layers);
                future_grad_output = sum(future_grad .* obj.output);
                
                obj.grad{1} = future_grad .* obj.a;
                obj.grad{1} = obj.grad{1} - bsxfun(@times, obj.a, future_grad_output);
            end
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end