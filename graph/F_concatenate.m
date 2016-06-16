function output = F_concatenate(prev_layers)

for i=1:length(prev_layers)
    prev_a{i} = prev_layers{i}.a;
end
output = cell2mat(prev_a');

end
