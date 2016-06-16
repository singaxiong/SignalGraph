% input is of dimension dim x nFr, where dim is |V| * context size and |V|
% is the vocabulary size. 
function output = F_word2vec(input, W, singlePrecision)
[dim, nFr, nSeg] = size(input);
if nSeg>1
    input = reshape(input, dim, nFr*nSeg);
end

[m,n] = size(W);

context = dim/n;

nVec = nFr*nSeg;

if 1
    curr_input = reshape(input, n, context * nVec);
    nonzero_col= sum(curr_input,2)>0;
    input_nonzero = full(curr_input(nonzero_col,:));
    if singlePrecision==1
        input_nonzero = single(input_nonzero);
    end
    output = W(:,nonzero_col) * input_nonzero;
    output = reshape(output, m*context, nVec);
else
    for i=1:context
        curr_input = input( (i-1)*n+1 : i*n, :);
        nonzero_col= sum(curr_input,2)>0;
        input_nonzero = full(curr_input(nonzero_col,:));
        if singlePrecision==1
            input_nonzero = single(input_nonzero);
        end
        output2{i} = W(:,nonzero_col) * input_nonzero;
    end
    output = cell2mat(output2');
end

if nSeg>1
    output = reshape(output, size(output,1), nFr, nSeg);
end

end
