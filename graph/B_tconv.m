function [grad, grad_W, grad_b] = B_tconv(prev_layer, curr_layer, future_layers, skip_grad)
input = prev_layer{1}.a;
[D,T,N] = size(input);
% Each row of W is a DxP filter to be applied along the time axis. We have
% totally H such filters.
W = curr_layer.W;
b = curr_layer.b;
[H,DP] = size(W);
P = DP/D;
halfP = (P-1)/2;

if 0
    % Instead of using conv2, we splice input and use matrix transform
    % We first put the N samples into a big sample, separated by zeros
    if strcmpi(class(input), 'gpuArray')
        input2 = gpuArray.zeros(D,T+P,N);
    else
        input2 = zeros(D,T+P,N);
    end
    input2(:,halfP+1:halfP+T,:) = input;
    X = reshape(input2, D, (T+P)*N);
    context = -halfP : halfP;
    X2 = ExpandContext_v2(X, context);
else
    X2 = curr_layer.X2;
end

X3 = reshape(X2, size(X2,1), size(X2,2)/N,N);

if isfield(curr_layer, 'range') && strcmpi(curr_layer.range, 'valid') == 0
    X3 = X3(:,halfP+1:halfP+T,:);
else
    X3 = X3(:,halfP*2+1:T,:);
end
X4 = reshape(X3, size(X3,1), size(X3,2)*N);

future_grad = GetFutureGrad(future_layers, curr_layer);
if strcmpi(class(future_grad), 'gpuArray')
    grad = gpuArray.zeros(D,T,N);
else
    grad = zeros(D,T,N);
end

fg = reshape(future_grad, H, size(future_grad,2)*N);

% Y2 = F_affine_transform(X2, W, b);
grad_W = fg * X4';
grad_b = sum(fg,2);

if skip_grad==0
    gradX4 = W' * fg;
    gradX4 = reshape(gradX4, size(gradX4,1), size(gradX4,2)/N, N);
    
    % now we need to distribute gradX4 to the input
    if strcmpi(class(input), 'gpuArray')
        gradX2 = gpuArray.zeros(size(X2,1), size(X2,2)/N,N);
    else
        gradX2 = zeros(size(X2,1), size(X2,2)/N,N);
    end
    if isfield(curr_layer, 'range') && strcmpi(curr_layer.range, 'valid') == 0
        gradX2(:,halfP+1:halfP+T,:) = gradX4;
    else
        gradX2(:,halfP*2+1:T,:) = gradX4;
    end
    gradX2 = reshape(gradX2, size(X2,1), size(X2,2));
    fl{1}.grad = gradX2;
    fl{1}.name = 'dummy';
    gradX = B_splice(fl, P);
    grad = reshape(gradX, D, T+P, N);
    grad = grad(:, halfP+1:halfP+T,:);
end

end
