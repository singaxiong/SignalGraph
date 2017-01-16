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

if N>1
    [validMask, variableLength] = getValidFrameMask(curr_layer);
    future_gradUtt = ExtractVariableLengthTrajectory(future_grad, validMask);
    YUtt = ExtractVariableLengthTrajectory(Y, validMask);
    X = reshape(X, D*C, T, N);
    XUtt = ExtractVariableLengthTrajectory(X, validMask);
    for i=1:N
        grad{i} = GetGradUtt(reshape(XUtt{i},D,C,size(XUtt{i},2)),YUtt{i},phi_s(:,:,:,:,i), phi_n(:,:,:,:,i), ...
            weight(:,:,i), lambda(:,:,:,:,i), future_gradUtt{i}, noiseCovL2);
    end
    grad = cell2mat_gpu(grad);
    grad = permute(grad, [1 3 2]);
else
    grad = GetGradUtt(X,Y,phi_s, phi_n, weight, lambda, future_grad, noiseCovL2);
end

end

%%
function grad = GetGradUtt(X,Y,phi_s, phi_n, weight, lambda, future_grad, noiseCovL2)
[D,C,T,N] = size(X);
u = zeros(C,1);
u(1) = 1;
for f=1:D
    x = squeeze(X(f,:,:));
    xx = x*x';
    ww = weight(f,:).' * conj(weight(f,:));
    phi_n_inv = inv(phi_n{1,1,f});
    yy = abs(Y(f,:).*conj(Y(f,:)));
    dyyy = sum(future_grad(f,:).*yy);
    dyxx = bsxfun(@times, x, future_grad(f,:)) * x';
    
    if 1
        grad_phi_s(:,:,f) = 2 * phi_n_inv * dyxx * weight(f,:).' * u' / lambda{1,1,f} -2 * dyyy * phi_n_inv / lambda{1,1,f};
        grad_phi_n(:,:,f) = - 2* phi_n_inv * dyxx * ww + 2 * dyyy * phi_n_inv * phi_s{1,1,f} * phi_n_inv / lambda{1,1,f};
    elseif 0    % no lambda
        grad_phi_s(:,:,f) = 2 * phi_n_inv * dyxx * weight(f,:).' * u';
        grad_phi_n(:,:,f) = - 2* phi_n_inv * dyxx * ww;
    else    % only lambda
        grad_phi_s(:,:,f) = -2 * dyyy * phi_n_inv / lambda{1,1,f};
        grad_phi_n(:,:,f) = 2 * dyyy * phi_n_inv * phi_s{1,1,f} * phi_n_inv / lambda{1,1,f};
    end
end

grad = conj([reshape(grad_phi_s,C*C*D,1); reshape(grad_phi_n,C*C*D,1)]);
end

