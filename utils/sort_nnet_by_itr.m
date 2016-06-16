function sorted_model_files = sort_nnet_by_itr(model_files)
if length(model_files)==1
    sorted_model_files = model_files;
    return;
end

for i=1:length(model_files)
    idx = regexp(model_files{i}, '\.itr');
    if length(idx)==0
        continue;
    end
    tmp = model_files{i}(idx+4:end);
    idx = regexp(tmp, '\.');
    itr(i) = str2num( tmp(1:idx(1)-1));
end

[tmp, idx] = sort(itr, 'descend');
sorted_model_files = model_files(idx);

end
