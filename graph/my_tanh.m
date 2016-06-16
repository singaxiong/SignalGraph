function output = tanh(input)
tmp = exp(-2*input);
output = (1-tmp) ./ (1+tmp);
end