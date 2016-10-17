% compute the joint cost between neighboring data points
%
function grad = B_jointCost(input_layer, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);

dimension = curr_layer.dimension;   % dimension defines along which dimension the smoothness are defined. 

precision = class(gather(input(1)));
if IsInGPU(input)
    grad = gpuArray.zeros(D,T,N, precision);
else
    grad = zeros(D,T,N, precision);
end

for di= 1:length(dimension)
    d = dimension(di);
    if d==1
        diff = input(2:end,:,:) - input(1:end-1,:,:);
        grad(1:end-1,:,:) = grad(1:end-1,:,:) - diff;
        grad(2:end,:,:) = grad(2:end,:,:) + diff;
    elseif d==2
        diff = input(:,2:end,:) - input(:,1:end-1,:);
        grad(:,1:end-1,:) = grad(:,1:end-1,:) - diff;
        grad(:,2:end,:) = grad(:,2:end,:) + diff;
    elseif d==3
        diff = input(:,:,2:end) - input(:,:,1:end-1);
        grad(:,:,1:end-1) = grad(:,:,1:end-1) - diff;
        grad(:,:,2:end) = grad(:,:,2:end) + diff;
    end
end

end