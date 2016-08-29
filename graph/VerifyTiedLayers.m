function [dimMismatch, isTranspose] = VerifyTiedLayers(tiedLayers)

for j=1:length(tiedLayers)
    dim(:,j) = tiedLayers{j}.dim;
end
dimMismatch = [];
isTranspose = [];
for j=2:length(tiedLayers)
    if sum(abs(dim(:,j)-dim(:,1)))
        if sum(abs(dim(end:-1:1,j)-dim(:,1)))
            dimMismatch(j) = 1;
            fprintf('Error: dimension mismatch between layers that share the same weight matrix\n');
        else
            isTranspose(j) = 1;
        end
    else
        isTranspose(j) = 0;
    end
end

end