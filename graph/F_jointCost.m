% compute the joint cost between neighboring data points
%
function cost = F_jointCost(input_layer, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);
dimension = curr_layer.dimension;   % dimension defines along which dimension the smoothness are defined. 

cost = 0;
if IsInGPU(input)
    cost = gpuArray.zeros(1,1);
end

for di= 1:length(dimension)
    d = dimension(di);
    if d==1
        diff = input(2:end,:,:) - input(1:end-1,:,:);
    elseif d==2
        diff = input(:,2:end,:) - input(:,1:end-1,:);
    elseif d==3
        diff = input(:,:,2:end) - input(:,:,1:end-1);
    end
    
    cost = cost + 0.5 * sum(sum(sum(diff.*diff)))/numel(diff);
end

end