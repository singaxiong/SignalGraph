
function classID = phoneID2classID(phoneID, vocab)

classID = single(zeros(size(phoneID)));

c = unique(phoneID);
for i = 1:length(c)
    pos = find(vocab==c(i));
    idx = phoneID==c(i);
    classID(idx) = pos;
end
end