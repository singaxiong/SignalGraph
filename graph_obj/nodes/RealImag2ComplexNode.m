% convert D-dimensional real-valued vectors to D/2-dimensional
% complex-valued vectors. The first D/2 elements are real part, while the
% second D/2 elements are imaginary part. 
%
classdef RealImag2ComplexNode < GraphNode
    properties
        dimIdx = 1;
    end
    methods
        function obj = RealImag2ComplexNode(dimOut, dimIdx)
            obj = obj@GraphNode('RealImag2Complex',dimOut);
            if nargin>=2
                obj.dimIdx = dimIdx;
            end
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            [D(1), D(2), D(3), D(4)] = size(input);

            permuteIdx = 1:4;
            permuteIdx(obj.dimIdx) = [];
            permuteIdx = [obj.dimIdx permuteIdx];
            input = permute(input, permuteIdx);
            
            realpart = input(1:D(obj.dimIdx)/2,:,:);
            imagpart = input(D(obj.dimIdx)/2+1:end,:,:);
            
            obj.a = realpart + sqrt(-1)*imagpart;
            
            [~, inversePermuteIdx] = sort(permuteIdx);
            obj.a = permute(obj.a, inversePermuteIdx);

            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            future_grad = obj.GetFutureGrad(future_layers);
            
            real_grad = real(future_grad);
            imag_grad = -imag(future_grad);
            obj.grad{1} = cat(obj.dimIdx, real_grad, imag_grad);
            
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
        
    end
    
end