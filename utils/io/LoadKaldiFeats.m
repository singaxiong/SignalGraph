function [uttID, feat] = LoadKaldiFeats(featRoot, files)

uttID = {};
feat = {};
for i=1:length(files)
    PrintProgress(i, length(files), 1, files{i});
    if length(regexp(files{i}, 'cmvn'))>0; continue; end
    [uttID_tmp, feat_tmp] = readKaldiFeature([featRoot '/' files{i}]);
    uttID = [uttID uttID_tmp];
    feat = [feat feat_tmp];
end
