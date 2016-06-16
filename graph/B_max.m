function grad = B_max(prev_layer, future_layers)
[D,T] = size(prev_layer{1}.a);
[~,max_idx] = max(prev_layer{1}.a, [], 2);

future_grad = 0;
for i=1:length(future_layers)
    future_grad = future_grad + future_layers{i}.grad;
end

if strcmpi(class(future_grad), 'gpuArray')
    grad = gpuArray.zeros(D,T);
else
    grad = zeros(D,T);
end


offset = (max_idx-1)*D;
idx = offset+ [1:D]';
grad(idx) = future_grad;

end
