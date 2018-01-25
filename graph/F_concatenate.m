function output = F_concatenate(prev_layers)

output = prev_layers{1}.a;
for i=2:length(prev_layers)
    output = [output; prev_layers{i}.a];
end

end
