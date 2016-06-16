function grad = B_real_imag2BFweight_beamforming_power(X, beamform_layer, after_power_layer, weight_layer, BF_frame_number)
% X is the multichannel complex spectrum inputs
[N,C,T] = size(X);
% Y is the beamforming's output
Y = beamform_layer.a;
% weight is the beamforming weight
weight = weight_layer.a;
% future_grad is the gradient of the power specturm of beamformed signal
future_grad = after_power_layer.grad;

if size(weight,2)==T
    online = 1;
else
    online = 0;
end

% real_grad = zeros(N,C);
% imag_grad = zeros(N,C);
% tic
% for t = 1:T
%     for k= 1:N
%         for c = 1:C
%             tmp = Y(k,t)*conj(X(c,t,k));
%             real_grad(k,c) = real_grad(k,c) + future_grad(k,t)*2*real(tmp);
%             imag_grad(k,c) = imag_grad(k,c) + future_grad(k,t)*2*imag(tmp);
%         end
%     end
% end
% toc;
% tic
% real_grad = zeros(N,C);
% imag_grad = zeros(N,C);
future_grad_Y = future_grad.*Y;
if 0
    for c = 1:C
        tmp = future_grad_Y.*squeeze(X(c,:,:))';
        tmp_real = real(tmp);
        tmp_imag = -imag(tmp);
        real_grad(:,c) = 2*sum(tmp_real(:,:),2);
        imag_grad(:,c) = 2*sum(tmp_imag(:,:),2);
    end
    grad = [reshape(real_grad,N*C,1); reshape(imag_grad,N*C,1)];
else
    X_tmp = permute(X, [1 3 2]);
    tmp = bsxfun(@times, conj(X_tmp), future_grad_Y);
    
    if online
        tmp = 2*permute(tmp, [1 3 2]);
        tmp = reshape(tmp, N*C,T);
        grad = [real(tmp); -imag(tmp)];
    else
        tmp = 2*reshape(sum(tmp,2), N*C,1);
        real_grad = real(tmp);
        imag_grad = -imag(tmp);
        grad = [real_grad; imag_grad];
        grad = repmat(grad/BF_frame_number, 1, BF_frame_number);
    end
end

end
