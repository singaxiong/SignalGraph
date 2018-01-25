classdef AffineNode < GraphNodeUpdatable
    
    methods
        function obj = AffineNode(dimOut)
            obj = obj@GraphNodeUpdatable('affine', dimOut);
        end
        
        function obj = initialize(obj, gauss, negbias, r)
            if sum(abs(size(obj.W)-obj.dim))>0
                if gauss
                    obj.W = 3/sqrt(obj.dim(2)) * randn(obj.dim);
                else
                    obj.W = rand(obj.dim) * 2 * r - r;
                end
            end
            if sum(abs(size(obj.b,1)-obj.dim(1)))>0
                if negbias
                    obj.b = rand(obj.dim(1),1)/5 - 4.1;
                else
                    obj.b = zeros(obj.dim(1),1);
                end
            end
        end
        
        function obj = forward(obj,prev_layers)
            input_layer = prev_layers{1};
            input = input_layer.a;
            
            [D,M,N] = size(input);
            if N==1
                obj.a = bsxfun(@plus, obj.W * input, obj.b);
                obj.mask = [];
            else
                [obj.mask] = getValidFrameMask(input_layer);
                input2 = reshape(input, D,M*N);
                output2 = bsxfun(@plus, obj.W * input2, obj.b);
                obj.a = reshape(output2, size(output2,1), M,N);
            end
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            input = prev_layers{1}.a;
            obj.mask = [];
            
            future_grad = GetFutureGrad(future_layers);
            
            [n1,n2,n3] = size(future_grad);
            if n3>1     % reshape the matrix to 2D
                [validFrameMask, variableLength] = getValidFrameMask(prev_layers{length(prev_layers)});
                if variableLength
                    future_grad = PadShortTrajectory(future_grad, validFrameMask, 0);
                end
                future_grad = reshape(future_grad, n1,n2*n3);
                input = reshape(input, size(input,1), n2*n3);
            end
            
            if obj.updateWeight
                if issparse(input)
                    visible_nonzero_idx = find(sum(abs(input),2)>0);
                    visible_nonzero = full(input(visible_nonzero_idx,:));
                    obj.grad_W = sparse(size(obj.W,1),size(obj.W,2));
                    obj.grad_W(:,visible_nonzero_idx) = future_grad * visible_nonzero';
                else
                    obj.grad_W = conj(future_grad * input');
                end
                if ~isempty(obj.mask)
                    obj.grad_W = obj.grad_W .* obj.mask;
                end
            end
            if obj.updateBias
                obj.grad_b = sum(future_grad,2);
            else
                obj.grad_b = future_grad(:,1)*0;
            end
            if ~obj.skipGrad
                obj.grad = obj.W' * future_grad;
                if n3>1
                    obj.grad = reshape(obj.grad, size(obj.grad,1), n2, n3);
                end
            end
            
        end
        
    end
    
end