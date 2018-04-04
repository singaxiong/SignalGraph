% append delta and acceleration features in speech recognition
% generally used to capture temporal structure of features.
classdef DeltaNode < GraphNode
    properties
        
    end
    
    methods
        function obj = DeltaNode(dimIn)
            dimOut = 3*dimIn;
            obj = obj@GraphNode('Delta',dimOut);
            obj.dim(3:4) = obj.dim(1:2);
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            [D,D2,T,N] = size(input);
            input = reshape(input, D*D2, T, N);
                       
            if obj.variableLength
                input = obj.PadShortTrajectory(input, 'last'); 
                input2 = ExtractVariableLengthTrajectory(input);
                output = obj.AllocateMemoryLike([D*3,T,N], input);
                for i=1:N
                    output(:,1:size(input2{i},2),i) = comp_dynamic_feature(input2{i}',2,2)';
                end
                obj.a = output;
            else
                input2 = reshape(permute(input, [1 3 2]), D*D2*N,T);
                output2 = comp_dynamic_feature(input2',2,2)';
                output2 = reshape(output2, D, D2, N, 3, T);
                obj.a = reshape(permute(output2, [1 4 2 5 3]), D*3,D2,T,N);
            end
            
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            fgrad = obj.GetFutureGrad(future_layers);            
            [dim,D2,nFr,nSeg] = size(fgrad);
            if D2>1
                fgrad = reshape(fgrad, dim/3, 3, D2, nFr, nSeg);
                fgrad = permute(fgrad, [1 3 2 4 5]);
            end
            fgrad = reshape(fgrad, dim*D2, nFr, nSeg);
                
            dimS = dim/3;
            precision = class(gather(fgrad(1)));
            D = genDeltaTransform(nFr, 2, 1, precision);
            A = D*D;
            D = full(D);
            A = full(A);
            
            if nSeg==1
                obj.grad{1} = fgrad(1:dimS,:) + fgrad(dimS+1:dimS*2,:)*D + fgrad(dimS*2+1:end,:)*A;
            else
                if obj.variableLength
                    fgrad = PadShortTrajectory(fgrad, 0);
                    future_grad2 = ExtractVariableLengthTrajectory(fgrad);
                    for i=1:nSeg
                        nFrUtt = size(future_grad2{i},2);
                        D = genDeltaTransform(nFrUtt, 2, 1, precision);
                        A = D*D;
                        D = full(D);
                        A = full(A);
                        obj.grad{1}(:,1:nFrUtt,i) = future_grad2{i}(1:dimS,1:nFrUtt) + future_grad2{i}(dimS+1:dimS*2,1:nFrUtt)*D + future_grad2{i}(dimS*2+1:end,1:nFrUtt)*A;
                    end
                else
                    fgrad2 = permute(fgrad, [1 3 2]);
                    fgrad2S = reshape(fgrad2(1:dimS,:,:), dimS*nSeg,nFr);
                    fgrad2D = reshape(fgrad2(dimS+1:dimS*2,:,:), dimS*nSeg,nFr);
                    fgrad2A = reshape(fgrad2(dimS*2+1:end,:,:), dimS*nSeg,nFr);
                    obj.grad{1} = fgrad2S + fgrad2D*D + fgrad2A*A;
                    obj.grad{1} = reshape(obj.grad{1}, dimS, nSeg, nFr);
                    obj.grad{1} = permute(obj.grad{1}, [1 3 2]);
                end
            end
            obj.grad{1} = reshape(obj.grad{1}, [dimS, D2, nFr, nSeg]);
            
            obj = backward@GraphNode(obj, prev_layers, future_layers);
        end
    end
end