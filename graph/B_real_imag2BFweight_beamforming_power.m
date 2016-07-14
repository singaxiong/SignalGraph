function grad = B_real_imag2BFweight_beamforming_power(X, beamform_layer, after_power_layer, weight_layer, real_imag_weight)
% X is the multichannel complex spectrum inputs
[N,C,T,nSent] = size(X);
% Y is the beamforming's output
Y = beamform_layer.a;
% weight is the beamforming weight
weight = weight_layer.a;
% future_grad is the gradient of the power specturm of beamformed signal
future_grad = after_power_layer.grad;
online = size(weight,2)==T;

if nSent==1
    grad = BP_single_sentence(X, Y, future_grad, online, size(real_imag_weight,2));
else
    [mask] = getValidFrameMask(after_power_layer);
    future_grad2 = ExtractVariableLengthTrajectory(future_grad, mask);
    X2 = ExtractVariableLengthTrajectory(reshape(X,N*C,T,nSent), mask); 
    Y2 = ExtractVariableLengthTrajectory(Y, mask);
    maskWeight = CheckTrajectoryLength(real_imag_weight);
    for i=1:nSent
        grad{i} = BP_single_sentence(reshape(X2{i},N,C,size(X2{i},2)), Y2{i}, future_grad2{i}, online, sum(maskWeight(:,i)==0));
    end
    grad = cell2mat_gpu(grad);
    grad = PadGradientVariableLength(grad, maskWeight);
end

end

%%
function grad = BP_single_sentence(X, Y, future_grad, online, BF_frame_number)
[N,C,T] = size(X);
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