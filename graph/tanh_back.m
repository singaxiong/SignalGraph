function y = tanh(x)
exp_term = exp(x);
exp_term2 = exp(-x);
y = (exp_term - exp_term2) ./ (exp_term + exp_term2);

end

