function output = Tensor2MatWithBuffer(input, buffer_size, replicateBoundary)
if nargin<3
    replicateBoundary = 0;
end

[D,T,N] = size(input);

halfB = floor((buffer_size-1)/2);

if strcmpi(class(input), 'gpuArray')
    input2 = gpuArray.zeros(D,T+buffer_size,N);
else
    input2 = zeros(D,T+buffer_size,N);    
end
input2(:,halfB+1:halfB+T,:) = input;
if replicateBoundary
    input2(:,1:halfB,:) = repmat(input(:,1,:), [1 halfB 1]);
    input2(:,halfB+T+1:end,:) = repmat(input(:,end,:), [1 halfB+1 1]);
end
output = reshape(input2, D, (T+buffer_size)*N);

end