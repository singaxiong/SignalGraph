function seg = DivideSent2Segments(sent, seglen, segshift, outputCell)
if nargin<4
    outputCell = 1;
end
precision = class(gather(sent(1,1,1)));
[D, nFr] = size(sent);

nSeg = floor((nFr-seglen)/segshift) + 1;
if outputCell
    seg = cell(nSeg,1);
else
    seg = zeros(D, seglen, nSeg, precision);
end
for i=1:nSeg
    curr_seg = sent(:, (i-1)*segshift+1 : (i-1)*segshift + seglen );
    if outputCell
        seg{i} = curr_seg;
    else
        seg(:,:,i) = curr_seg;
    end
end

end
