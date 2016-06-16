
function grad = B_relu(future_layers, output)
mask = output>0;
if strcmpi(class(output), 'gpuArray')
    grad = gpuArray.zeros(size(output));
else
    grad = zeros(size(output));
end
for i=1:length(future_layers)
    grad(mask) = grad(mask) + future_layers{i}.grad(mask);
end
end
