function grad = B_MVDR_spatialCov(X, curr_layer, beamform_layer, after_power_layer)

% X is the multichannel complex spectrum inputs
[N,C,T,nSent] = size(X);
% weight is the beamforming weight
weight = reshape(curr_layer.a, N,C);
lambda = curr_layer.lambda;
phi_s = curr_layer.phi_s;
phi_n = curr_layer.phi_n;

% Y is the beamforming's output
Y = beamform_layer.a;
% future_grad is the gradient of the power specturm of beamformed signal
future_grad = after_power_layer.grad;

u = zeros(C,1);
u(1) = 1;
for f=1:N
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

grad = conj([reshape(grad_phi_s,C*C*N,1); reshape(grad_phi_n,C*C*N,1)]);

end