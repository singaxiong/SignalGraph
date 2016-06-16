% This function will find all subfolders of a directory
function subfolders = findSubfolders(rootdir)

[list ISDIR]= my_dir(rootdir);
subfolders = {};

for i=1:length(list)
    tmp = [rootdir '/' list{i}];
    if ISDIR(i)==1
        subfolders{end+1} = tmp;
        subsubfolders = findSubfolders(subfolders{end});
        subfolders = [subfolders subsubfolders];
    end
end
