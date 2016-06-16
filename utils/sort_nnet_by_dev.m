function sorted_model_files = sort_nnet_by_dev(model_files)

for i=1:length(model_files)
    idx = regexp(model_files{i}, '\.CV');
    CV_score(i) = str2num( model_files{i}(idx(end)+3:end-4) );
end

[tmp, idx] = sort(CV_score);
sorted_model_files = model_files(idx);

end
