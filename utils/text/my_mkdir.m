
% This function extend mkdir. it first check whether a folder is already
% exist.
function my_mkdir(dirname)

if exist(dirname, 'dir')==0
    mkdir(dirname)
end
