function kwslist = readKWS_xml(filename)
FID = fopen(filename);
currLine = fgetl(FID);
if regexp(currLine, 'xml version')
    currLine = fgetl(FID);
end

kwslist.kwid = {};
kwslist.file_name = {};
kwslist.channel = {};
kwslist.tbeg = {};
kwslist.dur = {};
kwslist.score = {};
kwslist.decision = {};

lineCount = 0;
while 1
    lineCount = lineCount + 1;
    PrintProgress(lineCount, 10^5, 50000);
    
    currLine = fgetl(FID);
    if currLine == -1
        break;
    elseif strcmpi(currLine, '</stdlist>') || strcmpi(currLine, '</kwslist>')
        break;
    end
    
    idx = regexp(currLine, '"');
    
    if length(regexp(currLine, 'detected_kwlist'))>0
        if length(regexp(currLine, '</detected_kwlist>'))==0
            curr_keyword_id = currLine(idx(3)+1:idx(4)-1);
        end
    elseif length(regexp(currLine, 'detected_termlist'))>0
        if length(regexp(currLine, '</detected_termlist>'))==0
            curr_keyword_id = currLine(idx(1)+1:idx(2)-1);
        end        
    else
        kwslist.kwid{end+1} = curr_keyword_id;
        kwslist.file_name{end+1} = currLine(idx(1)+1:idx(2)-1);
        kwslist.channel{end+1} = currLine(idx(3)+1:idx(4)-1);
        kwslist.tbeg{end+1} = currLine(idx(5)+1:idx(6)-1);
        kwslist.dur{end+1} = currLine(idx(7)+1:idx(8)-1);
        kwslist.score{end+1} = currLine(idx(9)+1:idx(10)-1);
        kwslist.decision{end+1} = currLine(idx(11)+1:idx(12)-1);
    end
end

fclose(FID);