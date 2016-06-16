function kwlist = LoadKWlist(filename)

matfile = [filename '.mat'];
if exist(matfile, 'file')
    load(matfile);
    return;
end

fprintf('Reading KWlist from XML file\n');
kwlist = readKWSlist_fast(filename);

fprintf('Converting strings into numbers\n');
for i=1:length(kwlist.kwid)
    PrintProgress(i, length(kwlist.kwid), 300);
    kwlist.dur{i} = str2double(kwlist.dur{i});
    kwlist.score{i} = str2double(kwlist.score{i});
    kwlist.tbeg{i} = str2double(kwlist.tbeg{i});
    if kwlist.verbose
        kwlist.raw_score{i} = str2double(kwlist.raw_score{i});
        kwlist.threshold{i} = str2double(kwlist.threshold{i});
    end        
end

save(matfile, 'kwlist');

end