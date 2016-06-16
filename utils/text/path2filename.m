function fileNames = path2filename(listFile, extensionLength)

if nargin<2
    extensionLength = 3;
end

fileNames = my_cat(listFile);
for i=1:length(fileNames)
    idx = regexp(dos2unix(fileNames{i}), '/');
    fileNames{i} = fileNames{i}(idx(end)+1:end-extensionLength-1);
end
