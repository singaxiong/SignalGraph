function grad = B_MVDR_spatialCov(X, curr_layer, beamform_layer, after_power_layer)

% X is the multichannel complex spectrum inputs
[D,C,T,N] = size(X);
% weight is the beamforming weight
weight = reshape(curr_layer.a, D,C,N);
lambda = curr_layer.lambda;
phi_s = curr_layer.phi_s;
phi_n = curr_layer.phi_n;

if isfield(curr_layer, 'noiseCovL2')
    noiseCovL2 = curr_layer.noiseCovL2;
else
    noiseCovL2 = 0;  % add noiseCovRegularization*\lambda*I to noise covariance, where \lambda is the maximum eigenvalue
end

% Y is the beamforming's output
Y = beamform_layer.a;
% future_grad is the gradient of the power specturm of beamformed signal
future_grad = after_power_layer.grad;
future_layers{1} = beamform_layer;
future_grad2 = GetFutureGrad(future_layers, curr_layer);

if N>1
    [validMask, variableLength] = getValidFrameMask(curr_layer);
    future_gradUtt = ExtractVariableLengthTrajectory(future_grad, validMask);
    YUtt = ExtractVariableLengthTrajectory(Y, validMask);
    X = reshape(X, D*C, T, N);
    XUtt = ExtractVariableLengthTrajectory(X, validMask);
    for i=1:N
        grad{i} = GetGradUtt(reshape(XUtt{i},D,C,size(XUtt{i},2)),YUtt{i},phi_s(:,:,:,:,i), phi_n(:,:,:,:,i), ...
            weight(:,:,i), lambda(:,:,:,:,i), future_gradUtt{i}, future_grad2(:,:,i), noiseCovL2);
    end
    grad = cell2mat_gpu(grad);
    grad = permute(grad, [1 3 2]);
else
    grad = GetGradUtt(X,Y,phi_s, phi_n, weight, lambda, future_grad, future_grad2, noiseCovL2);
end

end

%%
function grad = GetGradUtt(X,Y,phi_s, phi_n, weight, lambda, future_grad, future_grad2, noiseCovL2)
[D,C,T,N] = size(X);
u = zeros(C,1);
u(1) = 1;
future_grad2 = reshape(future_grad2, D,C);
if IsInGPU(X(1))
    grad_phi_s = gpuArray.zeros(C,C,D);
    grad_phi_n = gpuArray.zeros(C,C,D);    
else
    grad_phi_s = zeros(C,C,D);
    grad_phi_n = zeros(C,C,D);
end

for f=1:D
%     x = squeeze(X(f,:,:));
%     xx = x*x';
%     ww = weight(f,:).' * conj(weight(f,:));
    phi_n_inv = inv(phi_n{1,1,f});
%     yy = abs(Y(f,:).*conj(Y(f,:)));
%     dyyy = sum(future_grad(f,:).*yy);
%     dyxx = bsxfun(@times, x, future_grad(f,:)) * x';
    
    if 1
%         grad_phi_s(:,:,f) = 2 * phi_n_inv * dyxx * weight(f,:).' * u' / lambda{1,1,f} -2 * dyyy * phi_n_inv / lambda{1,1,f};
%         grad_phi_n(:,:,f) = - 2* phi_n_inv * dyxx * ww + 2 * dyyy * phi_n_inv * phi_s{1,1,f} * phi_n_inv / lambda{1,1,f};
        
            for ii = 1:C
                grad_phi_s(:,:,f) = grad_phi_s(:,:,f) + conj(future_grad2(f,ii)) *  phi_n_inv(ii,:)' * u' /lambda{1,1,f}...
                    - conj(future_grad2(f,ii)) * conj(weight(f,ii)) * phi_n_inv'/lambda{1,1,f};
                
                tmp = zeros(C,C);
                tmp(ii,:) = u'*phi_s{1,1,f}';
                grad_phi_n(:,:,f) = grad_phi_n(:,:,f) - conj(future_grad2(f,ii)) *  phi_n_inv' * tmp * phi_n_inv' / lambda{1,1,f}...
                    + conj(future_grad2(f,ii)) * conj(weight(f,ii)) * (phi_n_inv * phi_s{1,1,f} * phi_n_inv)' / lambda{1,1,f};
            end
            
    elseif 0    % no lambda
%         grad_phi_s(:,:,f) = 2 * phi_n_inv * dyxx * weight(f,:).' * u';
%         grad_phi_n(:,:,f) = - 2* phi_n_inv * dyxx * ww;
        
        if 0
            grad_phi_s2(:,:,f) = zeros(C,C);
            for ii = 1:C
                grad_phi_s2(:,:,f) = grad_phi_s2(:,:,f) + conj(future_grad2(f,ii)) *  phi_n_inv(ii,:)' * u';
            end
        else
            tmp = bsxfun(@times, permute(phi_n_inv', [1 3 2]), u');
            grad_phi_s(:,:,f) = squeeze(sum(bsxfun(@times, permute(conj(future_grad2(f,:)), [3 1 2]), tmp),3));
        end
        
        if 1
            grad_phi_n(:,:,f) = zeros(C,C);
            for ii = 1:C
                tmp = zeros(C,C);
                tmp(ii,:) = u'*phi_s{1,1,f}';
                grad_phi_n(:,:,f) = grad_phi_n(:,:,f) - conj(future_grad2(f,ii)) *  phi_n_inv' * tmp * phi_n_inv';
            end
        else
            
        end
        
        
        
    else    % only lambda
%         grad_phi_s(:,:,f) = -2 * dyyy * phi_n_inv / lambda{1,1,f};
%         grad_phi_n(:,:,f) = 2 * dyyy * phi_n_inv * phi_s{1,1,f} * phi_n_inv / lambda{1,1,f};
        
        if 1
            grad_phi_s(:,:,f) = zeros(C,C);
            for ii = 1:C
                grad_phi_s(:,:,f) = grad_phi_s(:,:,f) - conj(future_grad2(f,ii)) *  phi_n_inv'/lambda{1,1,f}^2;
            end
        else
            
        end
        if 1
            grad_phi_n(:,:,f) = zeros(C,C);
            for ii = 1:C
                grad_phi_n(:,:,f) = grad_phi_n(:,:,f) + conj(future_grad2(f,ii)) *  (phi_n_inv * phi_s{1,1,f} * phi_n_inv)' / lambda{1,1,f}^2;
            end
        else
            
        end
    end
end

grad = conj([reshape(grad_phi_s,C*C*D,1); reshape(grad_phi_n,C*C*D,1)]);
end

