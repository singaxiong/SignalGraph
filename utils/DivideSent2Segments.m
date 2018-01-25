function seg = DivideSent2Segments(sent, seglen, segshift, outputCell, tail2add)
if nargin<5
    tail2add = 0; % this is the amount we want to add to the end of each segment. Usually used to account for the overlapped frames within segments
end
if nargin<4
    outputCell = 1;
end

if ischar(sent)
    if isempty(regexp(sent,'\t', 'once'))
        terms = strsplit(sent);
        filename = terms{1};
        startTime = str2num(terms{2});
        stopTime = str2num(terms{3});
        duration = stopTime-startTime;
        
        nSeg = floor((duration-seglen)/segshift) + 1;
        seg = cell(nSeg,1);
        for i=1:nSeg
            seg{i} = [filename ' ' num2str((i-1)*segshift+startTime) ' ' num2str((i-1)*segshift + startTime + seglen -1 )];
        end
    else
        words = strsplit(sent, '\t');
        filename = words{1};
        terms = strsplit(words{2}, '=');
        sampleRange = strsplit(terms{2}, ',');
        startTime = str2num(sampleRange{1});
        stopTime = str2num(sampleRange{2});
        duration = stopTime-startTime+1;    % note we do not support second as unit of time. only support sample as unit of time.

        nSeg = floor((duration-seglen-tail2add)/segshift) + 1;
        seg = cell(nSeg,1);
        for i=1:nSeg
            seg{i} = [filename sprintf('\tsample=') num2str((i-1)*segshift+startTime) ',' num2str((i-1)*segshift + startTime + seglen -1 +tail2add)];
        end
    end
else
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


end
