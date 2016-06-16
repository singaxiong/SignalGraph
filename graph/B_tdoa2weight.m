function grad = B_power_spectrum(future_grad, curr_layer, tdoa)
% we don't implement this function now, as it will be used with
% beamforming. We will implement the joint gradient of Power spectrum and
% frequency domain beamforming. 
[N,C] = size(future_grad);
omega = curr_layer.freqBin;
weight = curr_layer.a;

for i=1:C
    grad(i) = -sqrt(-1)*omega*(weight(:,i).*future_grad(:,i));
end
grad = real(grad);

T = size(tdoa,2);
grad = repmat(grad', 1, T)/T;
end
