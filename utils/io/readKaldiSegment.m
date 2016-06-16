function [convID, segment] = readKaldiSegment(segmentFile)

lines = my_cat(segmentFile);

for i=1:length(lines)
    tmp = textscan(lines{i}, '%s %s %f %f',1);
    segmentID = tmp{1}{1};
    currConvID = tmp{2}{1};
    time1 = tmp{3};
    time2 = tmp{4};

    if i==1     % first line
        convID{1} = currConvID;
        segment{1}.time1(1) = time1;
        segment{1}.time2(1) = time2;
        segment{1}.ID{1} = segmentID;
%         fprintf('Now reading conversation %s\n',convID{end});
    else
        if strcmp(currConvID, convID{end})==0  % new conversation
            convID{end+1} = currConvID;
            segment{end+1}.time1(1) = time1;
            segment{end}.time2(1) = time2;
            segment{end}.ID{1} = segmentID;
%             fprintf('Now reading conversation %s\n',convID{end});
        else
            segment{end}.time1(end+1) = time1;
            segment{end}.time2(end+1) = time2;            
            segment{end}.ID{end+1} = segmentID;
        end
    end
end
end