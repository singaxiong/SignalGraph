function output = F_min(input_layer, curr_layer)
input = input_layer.a;
if isfield(curr_layer, 'idx')
    output = min(input, [], curr_layer.idx);
else
    output = min(input,[], 2);
end
end
