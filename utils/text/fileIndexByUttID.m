function fileIndex = fileIndexByUttID(filelist)

for i=1:length(filelist)
    [~,uttID] = fileparts(filelist{i});
    fileIndex.(['U_' regexprep(uttID, '-', '_')]) = filelist{i};
end

end
