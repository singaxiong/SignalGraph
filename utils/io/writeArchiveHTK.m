% Save features into block of utterances
function writeArchiveHTK(list, feat, archive_name, block_size)
nUtt = length(feat);
if nargin < 4
    block_size = 1000;
end
dim = size(feat{1},2);

blk_idx = 1;
for i=1:block_size:nUtt
    partID = sprintf('.%d', blk_idx);
    idx = min(nUtt, i+block_size-1);
    tmpFeat = feat(i:idx);
    tmpList = list(i:idx);

    clear nFr
    for j=1:length(tmpFeat)
        nFr(j) = size(tmpFeat{j}, 1);
    end
    
    tmpFeat = cell2mat(tmpFeat');
    writeHTK([archive_name partID '.ark'], tmpFeat, 'MFCC_0_D_A');
    my_dump([archive_name partID '.scp'], tmpList);
    save([archive_name partID '.nFr.mat'], 'nFr');
    blk_idx = blk_idx + 1;
end
