% Convert input file path to windows or unix format, depending on the
% current machine type. 
% Assumption: the drive letter "F:/" will be mapped to "/F/" and vice
% versa. That is, the same drive is shared by unix and windows machines,
% and their difference is only the way to represent the drive letter. 
%
function output_path = FormatPath4OS(input_path, isFullPath)
if nargin<2
    if strcmpi(input_path, '.')
        isFullPath=0;
    else
        isFullPath=1;
    end
end

input_path2 = dos2unix(input_path);

if isFullPath==0
    % if the input path is partial, we only need to convert the slashes
    % direction
    output_path = input_path2;
else
    % if it is full path, we also need to use the correct drive letter
    % check whether the path is windows or linus style
    idx = regexp(input_path2, ':');
    if ispc     % the current computer is a windows
        if length(idx)>0    % the input path is windows style, no change
            output_path = input_path2;
        else    % change from unix style to windows style
            output_path = [input_path2(2) ':' input_path2(3:end)];
        end
    else
        if length(idx)>0    % change from windows style to unix style
            output_path = ['/' regexprep(input_path2, ':', '')];
        else    % no change
            output_path = input_path2;
        end
    end
    
end
