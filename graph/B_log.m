
function grad = B_log(future_layers, input,b)
grad = 0;
for i=1:length(future_layers)
	grad = grad + 1./(input+b).*future_layers{i}.grad;
end
end