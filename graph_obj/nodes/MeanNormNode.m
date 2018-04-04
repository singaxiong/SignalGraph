classdef MeanNormNode < GraphNode
    properties
        context = 0;
    end
    
    methods
        function obj = MeanNormNode(dimOut)
            obj = obj@GraphNode('MeanNorm',dimOut);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);

            input = prev_layers{1}.a;
            [D,T,N] = size(input);
            if N==1
                obj.a = CMN(input')';
            else
                if obj.variableLength
                    input2 = obj.ExtractVariableLengthTrajectory(input);
                    output = obj.AllocateMemoryLike([D T N], input);
                    for i=1:N
                        output(:,1:size(input2{i},2),i) = CMN(input2{i}')';
                    end
                    obj.a = output;
                else
                    input2 = reshape(permute(input, [1 3 2]), D*N,T);
                    output = CMN(input2')';
                    obj.a = permute(reshape(output, D, N, T), [1 3 2]);
                end
            end
            
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            future_grad = obj.GetFutureGrad(future_layers);
            
            [D,T,N] = size(future_grad);
            
            if N==1
                obj.grad = CMN(future_grad')';
            else
                if variableLength
                    future_grad2 = obj.ExtractVariableLengthTrajectory(future_grad);
                    for i=1:N
                        grad{i} = CMN(future_grad2{i}')';
                    end
                    grad = cell2mat_gpu(grad);
                    
                    obj.grad{1} = obj.PadGradientVariableLength(grad);
                else
                    future_grad2 = reshape(permute(future_grad, [1 3 2]), D*N,T);
                    grad = CMN(future_grad2')';
                    obj.grad{1} = permute(reshape(grad, D, N, T), [1 3 2]);
                end
            end
            
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
    end
end