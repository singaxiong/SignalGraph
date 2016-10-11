function posterior = phoneID2posterior_multisoftmax(phoneID, classID)
[dim, nFr] = size(phoneID);
phoneID2 = reshape(phoneID, dim*nFr,1);

nClass = length(classID);
nSample = length(phoneID);
posterior = single(zeros(nClass, nSample));


c = unique(phoneID2);
for i = 1:length(c)
    pos = find(classID==c(i));
    idx = phoneID2==c(i);
    posterior(pos,idx) = 1;
end
posterior = reshape(posterior, nClass*dim, nFr);
end