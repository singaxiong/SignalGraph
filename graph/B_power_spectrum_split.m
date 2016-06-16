function grad = B_power_spectrum_split(future_layers, input)
grad = 0;
% [D,T] = size(input);
% D = D/2;
% realpart = input(1:D,:);
% imagpart = input(D+1:end,:);

for i=1:length(future_layers)
	grad = grad + 2*input.*repmat(future_layers{i}.grad, 2, 1);
end
end
