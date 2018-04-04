classdef LinearMapper
    properties
        biasOnly = 0;
        L2weight = 0;
    end
    methods
        function obj = LinearMapper()
        end
        
        function [A,b,cost, recon] = trainMapperMSE(obj, visible, target)
            [dim_v, nSample_v] = size(visible);
            [dim_t, nSample_t] = size(target);
            if nSample_v~=nSample_t
                fprintf('Error: the number of training samples in visible and target are different\n'); return;
            end
            
           
            if dim_v==dim_t && obj.biasOnly>0   % use identity transform matrix and only estimate bias vector
                A = eye(dim_v);
                b = mean(target-visible,2);
                if biasonly==2
                    b = ones(dim_v,1) * mean(b);
                end
            else
                visible2 = bsxfun(@plus, visible, -mean(visible,2));
                target2 = bsxfun(@plus, target, -mean(target,2));
                R = visible2 * visible2' / nSample_v;
                p = target2 * visible2' / nSample_v;
                A = p * inv( R + eye(dim_v)*obj.L2weight );
                b = mean(target,2) - A * mean(visible,2);
            end
            
            %obj.VerifyGradient(double(visible), double(target), randn(size(A)), randn(size(b)));
            
            recon = bsxfun(@plus, A * visible, b);
            cost = obj.ComputeCost(visible, target, A, b);
        end
        
        function [W, b,cost, recon] = trainMapperMSE_GD(obj, visible, target)
            [dim_v, nSample_v] = size(visible);
            [dim_t, nSample_t] = size(target);
            if nSample_v~=nSample_t
                fprintf('Error: the number of training samples in visible and target are different\n'); return;
            end
            
            lr = 1e-1;
            W = randn(dim_t,dim_v)/10 + sqrt(-1)*randn(dim_t,dim_v)/10;
            b = zeros(dim_t,1);
            for itr = 1:5000
                itrCost = ComputeCost(obj, visible, target, W, b);
                [gradW, gradB] = ComputeGrad(obj, visible, target, W, b);
                W = W - lr * gradW;
                b = b - lr * gradB;
                if mod(itr,100)==0
                    %fprintf('Cost at itr %d is %f\n', itr, itrCost);
                end
            end
            
            recon = bsxfun(@plus, W * visible, b);
            cost = obj.ComputeCost(visible, target, W, b);
        end
        
        
        function cost = ComputeCost(obj, visible, target, W, b)
            projected = bsxfun(@plus, W * visible, b);
            cost = 0.5*sum(sum( (projected-target).*conj(projected-target) )) / size(projected,2);
        end

        function [gradW, gradB] = ComputeGrad(obj, visible, target, W, b)
            target2 = bsxfun(@plus, target, -b);
            [D,T] = size(visible);
            gradW = -target2 * visible' + W * (visible * visible');
            gradW = gradW / T;
            gradB = b - 2*mean(target - W*visible,2);
        end

        function VerifyGradient(obj, visible, target, W, b)
            [gradW, gradB] = ComputeGrad(obj, visible, target, W, b);
            epsilon = 1e-4;
            for i=1:numel(W)
                W(i) = W(i) + epsilon;
                cost2 = ComputeCost(obj, visible, target, W, b);
                
                W(i) = W(i) - 2*epsilon;
                cost1 = ComputeCost(obj, visible, target, W, b);
                
                numGradReal = (cost2-cost1)/2/epsilon;
                
                W(i) = W(i) + epsilon*sqrt(-1);
                cost2 = ComputeCost(obj, visible, target, W, b);
                
                W(i) = W(i) - 2*epsilon*sqrt(-1);
                cost1 = ComputeCost(obj, visible, target, W, b);
                
                numGradImag = (cost2-cost1)/2/epsilon;
                
                theoGrad = gradW(i);
                fprintf('[TheoGrad/NumGrad] = [%f %f, %f %f], diff = %f %f\n', real(theoGrad), imag(theoGrad), numGradReal, numGradImag, real(theoGrad) - numGradReal, imag(theoGrad) - numGradImag);
            end
        end
    end
end
