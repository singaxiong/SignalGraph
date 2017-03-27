% compute matrix multiply X * Y
%
function [grad] = B_matrix_multiply(input_layers, curr_layer, future_layers)
X = input_layers{1}.a;
Y = input_layers{2}.a;

% input1 and input2

[Dx,Tx,Nx] = size(X);
[Dy,Ty,Ny] = size(Y);
if Tx ~=Dy
    fprintf('Error: matrix size not match each other\n');
end

future_grad = GetFutureGrad(future_layers, curr_layer);

if Nx==1 && Ny==1
    gradX = conj(future_grad * Y');
    gradY = X' * future_grad;
else
    [validFrameMask, variableLength] = getValidFrameMask(curr_layer);
    if variableLength
        future_grad = PadShortTrajectory(future_grad, validFrameMask, 0);
    end
    
    if IsInGPU(X(1))
        gradX = gpuArray.zeros(size(X));
        gradY = gpuArray.zeros(size(Y));
    else
        gradX = zeros(size(X));
        gradY = zeros(size(Y));
    end
    
    if Nx==1 && Ny>1    %use the same X matrix for all Y matrices
        Y = PadShortTrajectory(Y, validFrameMask, 0);
        for i = 1:Ny
            gradX = gradX + conj(future_grad(:,:,i) * Y(:,:,i)');
            gradY(:,:,i) = X' * future_grad(:,:,i);
        end
    elseif Nx>1 && Ny==1    %use the same Y matrix for all X matrices
        X = PadShortTrajectory(X, validFrameMask, 0);
        for i = 1:Nx
            gradX(:,:,i) = conj(future_grad(:,:,i) * Y');
            gradY = gradY + X(:,:,i)' * future_grad(:,:,i);
        end
    elseif Nx>1 && Ny>1
        Y = PadShortTrajectory(Y, validFrameMask, 0);
        X = PadShortTrajectory(X, validFrameMask, 0);
        
        for i = 1:N
            gradX(:,:,i) = conj(future_grad(:,:,i) * Y(:,:,i)');
            gradY(:,:,i) = X(:,:,i)' * future_grad(:,:,i);
        end
    end
end
grad{1} = gradX;
grad{2} = gradY;
end