% read features from archives, each stored in HTK feature format
function [list, feat] = readArchiveHTK(archive_name)

feat = {};
list = {};
nFr = [];
uttCnt = 0;

for blk_idx=1:10^5
    partID = sprintf('.%d', blk_idx);
    fileName = [archive_name partID '.ark'];
    
    if exist(fileName, 'file')==0
        break;
    end
    
    tmpFeat = readHTK(fileName);
    tmpList = my_cat([archive_name partID '.scp']);
    list = [list; tmpList];
    load([archive_name partID '.nFr.mat']);
    
    offset = 0;
    for i=1:length(nFr)
        uttCnt = uttCnt + 1;
        feat{uttCnt} = tmpFeat(offset+1:offset+nFr(i),:);
        offset = offset + nFr(i);
    end
end

end
