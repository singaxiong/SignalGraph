function output = F_inner_product(input_layers)

input1 = input_layers{1}.a;
input2 = input_layers{2}.a;

output = sum(input1 .* input2);

end
