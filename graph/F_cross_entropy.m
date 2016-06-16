function [cost,acc] = F_cross_entropy(input_layers, CE_layer)
[nSeg, output, target] = prepareCostEvaluation(input_layers, CE_layer);
m = size(output,2);

[~, recogClass] = max(output);
% [~, trueClass] = max(target);
trueClass = target;
acc = sum(recogClass==trueClass)/m;

if 0
    target2 = zeros(size(output));
    for i=1:m
        target2(target(i),i) = 1;
    end
    product = log(output) .* target2;
    cost = -1/m * sum( sum( product ) );
    cost = -1/m * sum(product(:));
else
    dim = size(output,1);
    offset = 0:dim:m*dim-1;
    idx = offset+double(trueClass);     % critical: we need to use double for trueClass as single has limited precision and will fail for very larger numbers
    output2 = output(idx);
    cost = -1/m*sum(log(output2));
    
    % check if cost is nan. If any element in output2 is 0, cost will be
    % nan. We need to prevent this to make the training continue. 
    if isnan(cost)
        output2 = max(output2, eps);
    end
end

if 0
    imagesc(output);    hold on
    plot(target,'r');   hold off
    pause
end
end