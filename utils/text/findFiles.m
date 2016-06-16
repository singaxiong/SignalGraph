% This function will find all subfolders of a directory
function files = findFiles(rootdir, extension, relative_path)
if nargin<3
    relative_path = 0;
end
if nargin<2
    extension = '*';
end
[list ISDIR]= my_dir(rootdir);
files = {};

accept_all_extension = 0;
if extension == '*'
    accept_all_extension = 1;
end

for i=1:length(list)
    tmp = [rootdir '/' list{i}];
    if ISDIR(i)==1
        subfiles = findFiles(tmp, extension, 0);
        files = [files subfiles];
    else
        if relative_path==1
            currFile = list{i};
        else
            currFile = [rootdir '/' list{i}];
        end
        if accept_all_extension
            files{end+1} = currFile;
        else
            idx = regexp(list{i}, '\.');
            if length(idx)==0
                % no extension, skip
                continue;
            end
            curr_extension = list{i}(idx(end)+1:end);
            if strcmpi(curr_extension, extension)
                files{end+1} = currFile;
            end
        end
    end
end
