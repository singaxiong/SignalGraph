% compute matrix multiply X * Y
%
function [output, validFrameMask] = F_matrix_multiply(input_layers, curr_layer)
X = input_layers{1}.a;
Y = input_layers{2}.a;

[Dx,Tx,Nx] = size(X);
[Dy,Ty,Ny] = size(Y);
if Tx ~=Dy
    fprintf('Error: matrix size not match each other\n');
end

validFrameMask = [];
if Nx==1 && Ny==1
    output = X*Y;
elseif Nx==1 && Ny>1    %use the same X matrix for all Y matrices
    [validFrameMask, variableLength] = getValidFrameMask(input_layers{2});
    Y2 = reshape(Y, Dy, Ty*Ny);
    output = X*Y2;
    output = reshape(output, Dx, Ty, Ny);
elseif Nx>1 && Ny==1    %use the same Y matrix for all X matrices
    [validFrameMask, variableLength] = getValidFrameMask(input_layers{1});
    X2 = reshape(permute(Y, [1 3 2]), Dx*Nx, Tx);
    output = X2*Y;
    output = permute(reshape(output, Dx, Nx, Ty), [1 3 2]);    
elseif Nx>1 && Ny>1
    if Nx~=Ny
        fprintf('Error: Nx not equal to Ny - F_matrix_multiply\n'); return;
    end
    N = Nx;
    [validFrameMask, variableLength] = getValidFrameMask(input_layers{1});    
    % when both inputs contain multiple sequences, we expect the validFrameMask to be the same
    
    if IsInGPU(X(1))
        output = gpuArray.zeros(Dx, Ty, N);
    else
        output = zeros(Dx, Ty, N);
    end
    for i = 1:N
        output(:,:,i) = X(:,:,i) * Y(:,:,i);
    end
end
end