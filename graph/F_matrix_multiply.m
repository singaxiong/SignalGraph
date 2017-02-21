% compute matrix multiply X * Y
%
function [output, validFrameMask] = F_matrix_multiply(input_layers)
X = input_layers{1}.a;
Y = input_layers{2}.a;

% input1 and input2

[Dx,Tx,N] = size(X);
[Dy,Ty,N] = size(Y);
if Tx ~=Dy
    fprintf('Error: matrix size not match each other\n');
end

if N==1
    output = X*Y;
    validFrameMask = [];
else
    [validFrameMask, variableLength] = getValidFrameMask(input_layers{2});

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