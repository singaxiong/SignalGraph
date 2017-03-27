function CostLayer = F_mixture_mse(input_layers, CostLayer)
predicted{1} = input_layers{1}.a;
predicted{2} = input_layers{2}.a;
ref{1} = input_layers{3}.a;
ref{2} = input_layers{4}.a;

[D,T,N] = size(ref{1});

DEBUG = 0;

ref_idx = [];
for i=1:N
    
    diff1 = [predicted{1}(:,:,i)-ref{1}(:,:,i) predicted{2}(:,:,i)-ref{2}(:,:,i)];
    cost1 = 0.5/T * sum(sum( diff1 .* conj(diff1) ));   % support both real and complex numbers

    diff2 = [predicted{1}(:,:,i)-ref{2}(:,:,i) predicted{2}(:,:,i)-ref{1}(:,:,i)];
    cost2 = 0.5/T * sum(sum( diff2 .* conj(diff2) ));   % support both real and complex numbers
    
    if cost1>cost2
        ref_idx(:,i) = [2 1];   % record the pair information by remembering the reference index
        cost(i) = cost2;
    else
        ref_idx(:,i) = [1 2];
        cost(i) = cost1;
    end
    
    if DEBUG
        subplot(1,2,1); imagesc( [predicted{1}(1:257,:,i); ref{ref_idx(1,i)}(1:257,:,i)]); colorbar
        subplot(1,2,2); imagesc( [predicted{2}(1:257,:,i); ref{ref_idx(2,i)}(1:257,:,i)]); colorbar
%         subplot(2,2,1); imagesc(predicted{1}(1:257,:,i)); colorbar
%         subplot(2,2,2); imagesc(predicted{2}(1:257,:,i)); colorbar
%         subplot(2,2,3); imagesc(ref{ref_idx(1,i)}(1:257,:,i)); colorbar; 
%         subplot(2,2,4); imagesc(ref{ref_idx(2,i)}(1:257,:,i)); colorbar
        pause
    end
end

CostLayer.a = mean(cost);
CostLayer.ref_idx = ref_idx;

end