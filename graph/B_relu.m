
function grad = B_relu(future_layers, curr_layer)
output = curr_layer.a;
if isfield(curr_layer, 'threshold')
    threshold = curr_layer.threshold;
else
    threshold = 0;
end
mask = output>threshold;

if strcmpi(class(output), 'gpuArray')
    grad = gpuArray.zeros(size(output));
else
    grad = zeros(size(output));
end

for i=1:length(future_layers)
    grad(mask) = grad(mask) + future_layers{i}.grad(mask);
end
end
