function [layer_grad] = computeNumericalGradientLayer_tree2(layer, data, para, randomParaLoc)
for si=1:length(data)
    data{si} = double(data{si});
end
if nargin<4
    randomParaLoc = 1;  % randomly choose parameter for test
end

[cost_func,layer] = DNN_Cost10(layer, data, para, 1);

EPSILON = 10^(-4);
for i=1:length(layer)
    if isfield(layer{i},'W') && layer{i}.update
        [m,n] = size(layer{i}.W);
        layer_grad{i}.gradW_theo = zeros(m,n);
        layer_grad{i}.gradW_num = zeros(m,n);
        if randomParaLoc
            nPara = 5;
        else
            nPara = m*n;
        end
        for j=1:nPara
            if randomParaLoc
                if issparse(layer{i}.grad_W)
                    [nonzero_idx1, nonzero_idx2] = find(layer{i}.grad_W);
                    rand_idx = randperm(length(nonzero_idx1));
                    idx1 = nonzero_idx1(rand_idx(1));
                    idx2 = nonzero_idx2(rand_idx(1));
                else
                    idx1 = randperm(m); idx1 = idx1(1);
                    idx2 = randperm(n); idx2 = idx2(1);
                end
            else
                idx2 = ceil(j/m);
                idx1 = j-(idx2-1)*m;
            end
            
            init_val = layer{i}.W(idx1,idx2);
            
            layer{i}.W(idx1,idx2) = layer{i}.W(idx1,idx2) + EPSILON;
            [cost_func2] = DNN_Cost10(layer, data, para, 2);
            
            layer{i}.W(idx1,idx2) = layer{i}.W(idx1,idx2) - 2*EPSILON;
            [cost_func1] = DNN_Cost10(layer, data, para, 2);
            
            num_grad = gather((cost_func2.cost-cost_func1.cost))/2/EPSILON;
            
            layer{i}.W(idx1,idx2) = init_val;
            
            if isfield(layer{i}, 'grad_W')==0
                fprintf('Lyaer %d, %s has no grad_W\n', i, layer{i}.name);
            end
            
            theo_grad = gather(full(layer{i}.grad_W(idx1,idx2)));
            fprintf('Layer %d, W(%d,%d),[num_grad,theo_grad] = [%f, %f], diff=[%2.10f, %E]\n', ...
                i, idx1,idx2,num_grad,theo_grad,num_grad-theo_grad, (num_grad-theo_grad)/mean(abs([num_grad theo_grad])));
            
            layer_grad{i}.gradW_num(idx1,idx2) = num_grad;
            layer_grad{i}.gradW_theo(idx1,idx2) = theo_grad;
            
        end
    end
    if isfield(layer{i},'b') && layer{i}.update
        [m] = length(layer{i}.b);
        for j=1:3
            idx1 = randperm(m); idx1 = idx1(1);
            
            init_val = layer{i}.b(idx1);
            
            layer{i}.b(idx1) = layer{i}.b(idx1) + EPSILON;
            [cost_func2] = DNN_Cost10(layer, data, para, 2);
            
            layer{i}.b(idx1) = layer{i}.b(idx1) - 2*EPSILON;
            [cost_func1] = DNN_Cost10(layer, data, para, 2);
            
            num_grad = gather((cost_func2.cost-cost_func1.cost))/2/EPSILON;
            
            layer{i}.b(idx1) = init_val;
            
            fprintf('Layer %d, b(%d),[num_grad,theo_grad] = [%f, %f], diff=%f\n', ...
                i, idx1,num_grad,layer{i}.grad_b(idx1),num_grad-layer{i}.grad_b(idx1));
        end
    end
end
end
