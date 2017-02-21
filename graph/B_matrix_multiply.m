% compute matrix multiply X * Y
%
function [grad] = B_matrix_multiply(input_layers, curr_layer, future_layers)
X = input_layers{1}.a;
Y = input_layers{2}.a;

% input1 and input2

[Dx,Tx,N] = size(X);
[Dy,Ty,N] = size(Y);
if Tx ~=Dy
    fprintf('Error: matrix size not match each other\n');
end

future_grad = GetFutureGrad(future_layers, curr_layer);
[n1,n2,n3] = size(future_grad);
if n3>1     % reshape the matrix to 2D
    [validFrameMask, variableLength] = getValidFrameMask(input_layers{length(input_layers)});
    if variableLength
        future_grad = PadShortTrajectory(future_grad, validFrameMask, 0); 
    end
%     future_grad = reshape(future_grad, n1,n2*n3);
%     Y = reshape(Y, size(Y,1), n2*n3);
end

if N==1
    gradX = conj(future_grad * Y');
    gradY = X' * future_grad;
else
    if IsInGPU(X(1))
        gradX = gpuArray.zeros(size(X));
        gradY = gpuArray.zeros(size(Y));
    else
        gradX = zeros(size(X));
        gradY = zeros(size(Y));
    end
    for i = 1:N
        gradX(:,:,i) = conj(future_grad(:,:,i) * Y(:,:,i)');
        gradY(:,:,i) = X(:,:,i)' * future_grad(:,:,i);
    end
end
grad{1} = gradX;
grad{2} = gradY;
end