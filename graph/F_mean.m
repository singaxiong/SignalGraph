function output = F_mean(input_layer, curr_layer)
input = input_layer.a;
if isfield(curr_layer, 'pool_idx')
    output = mean(input, curr_layer.pool_idx);
else
    output = mean(input,2);
end
end

