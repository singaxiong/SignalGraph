function [output] = F_softmax(input_layer)
% Sometimes, output's element may get to inf when single precision is used.
% in such cases, we will have nan cross entropy. In such case, we can
% simply ignore the minibatch in training. 
input = input_layer.a;
output = exp(input);
recip_sum_a = 1./sum(output);
output = bsxfun(@times, output, recip_sum_a);

end
