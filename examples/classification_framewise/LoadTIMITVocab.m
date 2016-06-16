function vocab = LoadTIMITVocab(vocab_file, dataroot, type)

if exist(vocab_file, 'file')
    load(vocab_file);
else
    if strcmpi(type, 'phone')
        filelist = findFiles(dataroot, 'PHN');
    elseif strcmpi(type, 'word')
        filelist = findFiles(dataroot, 'WRD');
    end
    
    vocab = [];
    for i=1:length(filelist)
        PrintProgress(i, length(filelist), 100);
        label = my_cat(filelist{i});
        for j=1:length(label)
            words = ExtractWordsFromString_v2(label{j});
            label{j} = words{3};
        end
        vocab = [vocab; label];
        if mod(i,10)==0
            vocab = unique(vocab);
        end
    end
    save(vocab_file, 'vocab');
end
