% filtering and pooling over channel index. First dimension of input is
% nBin*nCh. First dimension of output is nBin. 
% 
classdef BeamformingNode < GraphNode
    properties
        nCh = 2;
    end
    methods
        function obj = BeamformingNode(dimOut, nCh)
            obj = obj@GraphNode('Beamforming',dimOut);
            obj.nCh = nCh;
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            
            input1 = input_layers{1}.a;
            input2 = input_layers{2}.a;
            
            [D(1) D(2) D(3) D(4)] = size(input);
            
            output = bsxfun(@times, input1, conj(input2));
            
            output = reshape(output, [dimOut obj.nCh D(2:end)]);
            
            obj.a = squeeze(sum(output,2));

            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            if obj.skipGrad || obj.skipBP
                return;
            end
            
            input1 = input_layers{1}.a;
            intpu2 = input_layers{2}.a;
            
            [D(1) D(2) D(3) D(4)] = size(input);
            [Dw,Tw,Nw] = size(weight);
            
            future_grad = GetFutureGrad(future_layers, curr_layer);
            future_grad = permute(future_grad, [1 2 4 3]);
            input = reshape(input, nBin,Di/nBin,Ti,Ni);
            
            input2 = permute(input, [1 3 2 4]);
            grad{1} = bsxfun(@times, conj(input2), future_grad);
            grad{1} = permute(grad{1}, [1 3 2 4]);
            grad{1} = reshape(grad{1}, Di,Ti,Ni);
            
            if Tw==1
                grad{1} = sum(grad{1},2);
            end
            
            grad{2} = [];


            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end