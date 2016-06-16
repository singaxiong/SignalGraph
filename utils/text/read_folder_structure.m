function files_out = read_folder_structure(root,files_in)

[curr_files,isdir] = my_dir(root);

files_out = files_in;
for i=1:length(curr_files)
    if isdir(i) == 1
        files_out = read_folder_structure([root '\' curr_files{i}],files_out);
    else
        files_out{length(files_out)+1} = [root '\' curr_files{i}];
    end
end
