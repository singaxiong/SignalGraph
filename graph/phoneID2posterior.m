
function posterior = phoneID2posterior(phoneID, nClass, classID)
if nargin<3
    classID = 1:nClass;
end
nSample = length(phoneID);
posterior = single(zeros(nClass, nSample));

if 1
    c = unique(phoneID);
    for i = 1:length(c)
        pos = find(classID==c(i));
        if length(pos)==0
            i
        end
        idx = phoneID==c(i);
        posterior(pos,idx) = 1;
    end
else
    for cid = 1:length(classID)
        idx = find(phoneID==classID(cid));
        if length(idx)>0
            posterior(cid,idx) = 1;
        end
    end
end

if sum(abs(sum(posterior)-1))~=0
    fprintf('phoneID2posterior: Error: sum of posterior not equal to 1\n');
end