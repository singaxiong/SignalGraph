function output = F_max(input_layer, curr_layer)
input = input_layer.a;
if isfield(curr_layer, 'idx')
    output = max(input, [], curr_layer.idx);
else
    output = max(input,[], 2);
end
end
