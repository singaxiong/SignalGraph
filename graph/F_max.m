function output = F_max(input_layer, curr_layer)
input = input_layer.a;
if isfield(curr_layer, 'pool_idx')
    output = max(input, [], curr_layer.pool_idx);
else
    output = max(input,[], 2);
end
end
