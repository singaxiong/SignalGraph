

function output = F_enframe(input, frame_len, frame_shift)

useGPU = strcmpi(class(input), 'gpuArray');

% do not use GPU for enframe
output = my_enframe(gather(input), frame_len, frame_shift);

if useGPU
    output = gpuArray(output);
end

end
