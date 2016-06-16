function [mfcMVN] = batchSideMVN(uttID, spkID, mfc)

for i=1:length(spkID)
    PrintProgress(i, length(spkID), 10);
    % Find out which sentences belong to current spkID
    belongTo = strfind(uttID, spkID{i});
    idx = [];
    for j=1:length(belongTo)
        if length(belongTo{j})>0
            idx(end+1) = j;
        end
    end
    
    [mfcMVN(idx), meanV, varV] = SideMVNcore(mfc(idx));
end