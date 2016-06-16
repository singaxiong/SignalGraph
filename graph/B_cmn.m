function grad = B_cmn(future_layer)
[dim, T] = size(future_layer{1}.grad);
if strcmpi(class(future_layer{1}.grad), 'gpuArray')
    grad = gpuArray.zeros(dim,T);
else
    grad = zeros(dim, T);
end
for li = 1:length(future_layer)
    grad = grad + CMN(future_layer{li}.grad')';
end
end
