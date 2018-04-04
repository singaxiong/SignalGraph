classdef AffineNode < GraphNodeUpdatable
    
    methods
        function obj = AffineNode(dimOut)
            obj = obj@GraphNodeUpdatable('Affine', dimOut);
        end
        
        function obj = initialize(obj, gauss, negbias, r)
            if sum(abs(size(obj.W)-obj.dim([1 3])))>0
                if gauss
                    obj.W = 3/sqrt(obj.dim(3)) * randn(obj.dim);
                else
                    obj.W = rand(obj.dim([1 3])) * 2 * r - r;
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
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            [D(1),D(2),D(3),D(4)] = size(input);
            input2 = reshape(input, D(1),prod(D(2:end)));
            output2 = bsxfun(@plus, obj.W * input2, obj.b);
            obj.a = reshape(output2, [size(output2,1) D(2:end)]);
            obj = forward@GraphNodeUpdatable(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            input = prev_layers{1}.a;
            
            future_grad = obj.GetFutureGrad(future_layers);
            
            [n1,D2,n2,n3] = size(future_grad);
            if n3>1     % reshape the matrix to 2D
                if obj.variableLength
                    future_grad = obj.PadShortTrajectory(future_grad, 0);
                end
            end
            future_grad = reshape(future_grad, n1,D2*n2*n3);
            input = reshape(input, size(input,1), D2*n2*n3);
            
            if obj.updateWeight
                if issparse(input)
                    visible_nonzero_idx = find(sum(abs(input),2)>0);
                    visible_nonzero = full(input(visible_nonzero_idx,:));
                    obj.gradW = sparse(size(obj.W,1),size(obj.W,2));
                    obj.gradW(:,visible_nonzero_idx) = future_grad * visible_nonzero';
                else
                    obj.gradW = conj(future_grad * input');
                end
                if ~isempty(obj.WMask)
                    obj.gradW = obj.gradW .* obj.WMask;
                end
            end
            if obj.updateBias
                obj.gradB = sum(future_grad,2);
            else
                % obj.gradB = future_grad(:,1)*0;
            end
            if ~obj.skipGrad
                obj.grad{1} = obj.W' * future_grad;
                obj.grad{1} = reshape(obj.grad{1}, size(obj.grad{1},1), D2, n2, n3);
            end
            
            obj = backward@GraphNodeUpdatable(obj, prev_layers, future_layers);

        end
        
    end
    
end