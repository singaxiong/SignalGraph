function replaced = my_replace(original, pattern1, pattern2)

replaced = cell(size(original));

for i=1:length(original)
    replaced{i} = regexprep(original{i}, pattern1, pattern2);
end
