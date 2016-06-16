function [convID, side, segment] = readCTM(ctmFile, threshold)

ctm = my_cat(ctmFile);
if nargin<2
    threshold = 2;  % if the silence between two words are longer than the threshold, set a sentence boundary between the two words.
end

for i=1:length(ctm)
    tmp = textscan(ctm{i}, '%s %s %f %f %s',1);
    currConvID = tmp{1}{1};
    currSide = tmp{2}{1};
    time1 = tmp{3};
    duration = tmp{4};
    label = tmp{5}{1};
    
    if i==1     % first line
        convID{1} = currConvID;
        side{1} = currSide;
        segment{1}.time1(1) = time1;
        segment{1}.time2 = [];
        segment{1}.label{1} = label;
        fprintf('Now reading conversation %s\n',convID{end});
    else
        if strcmp(currConvID, convID{end})==0 || strcmp(currSide, side{end})==0   % new conversation
            segment{end}.time2(end+1) = segment_stop;
            % new conversation
            convID{end+1} = currConvID;
            side{end+1} = currSide;
            segment{end+1}.time1(1) = time1;
            segment{end}.time2 = [];
            segment{end}.label{1} = label;
            fprintf('Now reading conversation %s\n',convID{end});
        else
            % decide whether to have a new segment
            isBoundary = 0;
            seg_duration = segment_stop - segment{end}.time1(end);
            if time1-segment_stop > threshold * max(0.3, min(2, 10 / seg_duration) ) % the longer the segment, the smaller the threshold
                isBoundary = 1;
            end
                
%             if time1-segment_stop>threshold
%                 isBoundary = 1;
%             elseif segment_stop-segment{end}.time1(end)>20 && time1-segment_stop>threshold/2    % if current segment is already very long, half the threshold
%                 isBoundary = 1;
%             end
            if isBoundary
                segment{end}.time2(end+1) = segment_stop;
                % new segment
                segment{end}.time1(end+1) = time1;
                segment{end}.label{end+1} = label;
            else
                segment{end}.label{end} = [segment{end}.label{end} ' ' label];
            end
        end
    end
    segment_stop = time1 + duration;
end
segment{end}.time2(end+1) = segment_stop;
end