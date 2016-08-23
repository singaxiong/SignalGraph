function output = F_permute(input_layer, curr_layer)
input = input_layer.a;
output = permute(input,curr_layer.idx);

end
