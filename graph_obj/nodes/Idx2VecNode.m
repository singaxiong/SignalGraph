% generate 1-hot vector from indexes.
% expect D2 = 1.
classdef Idx2VecNode < GraphNode
    properties
    end
    methods
        function obj = Idx2VecNode(dimOut)
            obj = obj@GraphNode('Idx2Vec',dimOut);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            
            [D(1),D(2),D(3),D(4)] = size(input);
            assert(D(2)==1, sprintf('%s:forward, Error: D(2) = %d is not equal to 1', D(2)));
            input = squeeze(input);
            vocabSize = curr_layer.dim(1);
            
            if obj.variableLength
                data = obj.ExtractVariableLengthTrajectory(input);
                input = cell2mat_gpu(data);
            else
                input = reshape(input, D(1),D(3)*D(4));
            end
            
            output = obj.AllocateMemoryLike(vocabSize, length(input), gather(input(1)));
            
            offset = (0:(length(input)-1)) * vocabSize;
            idx = double(offset) + double(input);   % a small bug, if offset is double and input is single, the idex will have problem.
            output(idx) = 1;

            if D(4)>1
                if obj.variableLength
                    output = obj.PadGradientVariableLength(output);
                else
                    output = reshape(output, vocabSize, D(3), D(4));
                end
                if IsInGPU(input)
                    output = gpuArray(output);
                end
            end
            obj.a = reshape(output, vocabSize, 1, D(3), D(4));
            
            
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            future_grad = GetFutureGrad(future_layers, obj);
            % Idx2Vec is usually used to generate 1-hot vector, no back
            % propagation is needed. 
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end