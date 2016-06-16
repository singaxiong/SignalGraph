function grad = B_filter(future_layers, input_layers)
weight_layer = input_layers{1};
data_layer = input_layers{2};

X = data_layer.a;
[D,nFr] = size(X);
W = weight_layer.a;
filter_len = length(W)/D;
W = reshape(W, filter_len, D);

half_context = (filter_len-1)/2;
idx = [ones(1,half_context) 1:nFr ones(1,half_context)*nFr];
X2 = X(:,idx);
% for t=1:filter_len
%     output = output + bsxfun(@times, X2(:,t:end-filter_len+t), W(end-t+1,:)');
% end

if strcmpi(class(future_layers{1}.grad), 'gpuArray')
    grad2 = gpuArray.zeros(size(X2));
else
    grad2 = zeros(size(X2));
end
future_grad = GetFutureGrad(future_layers);
for t=1:filter_len
    grad2(:,t:end-filter_len+t) = grad2(:,t:end-filter_len+t) + bsxfun(@times, future_grad, W(end-t+1,:)');
    
end
grad = grad2(:,half_context+1:end-half_context);
grad(:,1) = grad(:,1) + sum(grad2(:,1:half_context),2);
grad(:,end) = grad(:,end) + sum(grad2(:,end-half_context+1:end),2);

end