function grad = B_mixture_mse(input_layers, CostLayer)
predicted{1} = input_layers{1}.a;
predicted{2} = input_layers{2}.a;
ref{1} = input_layers{3}.a;
ref{2} = input_layers{4}.a;

[D,T,N] = size(ref{1});

ref_idx = CostLayer.ref_idx;

precision = class(gather(predicted{1}(1)));
if IsInGPU(predicted{1}(1))
    for i=1:length(predicted)
        grad{i} = gpuArray.zeros(size(predicted{1}), precision);
    end
else
    for i=1:length(predicted)
        grad{i} = zeros(size(predicted{1}), precision);
    end    
end

for i=1:N
    idx = ref_idx(:,i);
    for j=1:length(idx)
        diff = predicted{j}(:,:,i) - ref{idx(j)}(:,:,i);
        grad{j}(:,:,i) = diff / T / N;
    end
end

end