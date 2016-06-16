
function output_path = ChoosePath4OS(input_paths)
if ispc     % the current computer is a windows
    output_path = input_paths{1};
else
    output_path = input_paths{2};
end
end
