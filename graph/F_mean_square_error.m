function cost = F_mean_square_error(input_layers, useMahaDist, CostLayer)
[nSeg, output, target, scale] = prepareCostEvaluation(input_layers, CostLayer);
m = size(output,2);

diff = output - target;
if useMahaDist ==1  % in case 1, we use diagonal covariance matrix, which can be class-dependent
   cost = 0.5/m * sum(sum( diff.^2 .* cost_scale ));
elseif useMahaDist ==2  % in case 2, we use full covariance matrix, which can only be global and not class-dependent
   cost = 0.5/m * trace(diff' * cost_scale * diff);
else
   cost = 0.5/m * sum(sum( diff.*diff ));
end
end