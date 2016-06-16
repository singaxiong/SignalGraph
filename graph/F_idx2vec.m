function output = F_idx2vec(input, curr_layer, single_precision)
[n1,n2,n3] = size(input);
vocabSize = curr_layer.dim(1);

if n3>1
    input = reshape(input, n1,n2*n3);
end

if 1
    if single_precision
        output = zeros(vocabSize, length(input), 'single');
    else
        output = zeros(vocabSize, length(input), 'double');
    end
    
    offset = (0:(length(input)-1)) * vocabSize;
    idx = double(offset) + double(input);   % a small bug, if offset is double and input is single, the idex will have problem. 
    output(idx) = 1;
    
else
    output2 = sparse(vocabSize, length(input));
    for j=1:length(input)
        output2(input(j),j)= 1;
    end
end

if n3>1
    output = full(output);
    output = reshape(output, size(output,1), n2, n3);
    if single_precision
        output = single(output);
    end
    if strcmpi(class(input), 'gpuArray')
        output = gpuArray(output);
    end
end

end