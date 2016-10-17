function grad = B_tdoa2weight(future_layers, curr_layer)

future_grad = GetFutureGrad(future_layers, curr_layer);

[N,C, T] = size(future_grad);
omega= curr_layer.freqBin;
weight = curr_layer.a;
j = sqrt(-1);

if T==1
    for i=2:C
        grad(i-1) = sum(  -j * omega' .* (weight(:,i)) .* (future_grad(:,i)) );
    end
else
    for t=1:T
        for i=2:C
            grad(t,i-1) = sum(  -j * omega' .* (weight(:,i,t)) .* (future_grad(:,i,t)) );
        end
    end
end

grad = real(grad');  % only allow tdoa to be real numbers


end
