function output = F_filter(input_layers)
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

if strcmpi(class(X), 'gpuArray')
    output = gpuArray.zeros(size(X));
else
    output = zeros(size(X));
end
output2 = output;
if D<filter_len
    for i=1:D
        output(i,:) = filter(W(:,i), 1, X(i,:));
    end
else
    for t=1:filter_len
        output = output + bsxfun(@times, X2(:,t:end-filter_len+t), W(end-t+1,:)');
    end
end

end