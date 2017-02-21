function [files_cell,ISDIR] = my_dir(root)
files = dir(root);
if isempty(files)
    files_cell = {};
    ISDIR = [];
    return;
end

idx = regexp(root, '*');

if length(idx)==0
    startIdx=3;
else
    startIdx = 1;
end

field_list = fields(files(1));
isdir_idx = find(strcmp(field_list, 'isdir')==1);

files2 = struct2cell(files);
files_cell = files2(1,startIdx:end);
ISDIR = files2(isdir_idx,startIdx:end);
ISDIR = cell2mat(ISDIR);


% files_cell = {};
% ISDIR = [];
% for i=1:length(files)
%     if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
%         continue;
%     end
%     ISDIR(end+1) = files(i).isdir;
%     files_cell{end+1} = files(i).name;
% end



% files = strvcat(files.name);
% files = files(3:size(files,1),:);
% 
% files_cell = cell(size(files,1),1);
% for i=1:size(files,1)
%     idx = regexp(files(i,:), ' ');
%     if idx>0
%         files_cell{i} = files(i,1:idx-1);
%     else
%         files_cell{i} = files(i,:);
%     end
% end