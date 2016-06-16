function rttm = readRTTM(filename)
rttm.type = {};
rttm.file = {};
rttm.channel = {};
rttm.tbeg = {};
rttm.dur = {};
rttm.ortho = {};
rttm.stype = {};
rttm.spkr_name = {};
rttm.confidence = {};

FID = fopen(filename);
while 1
    currLine = fgets(FID);
    if currLine == -1
        break;
    end
    
    idx = find(currLine==' ');
    rttm.type{end+1}        = currLine(1:idx(1)-1);
    rttm.file{end+1}        = currLine(idx(1)+1:idx(2)-1);
    rttm.channel{end+1}     = currLine(idx(2)+1:idx(3)-1);
    rttm.tbeg{end+1}        = currLine(idx(3)+1:idx(4)-1);
    rttm.dur{end+1}         = currLine(idx(4)+1:idx(5)-1);
    rttm.ortho{end+1}       = currLine(idx(5)+1:idx(6)-1);
    rttm.stype{end+1}       = currLine(idx(6)+1:idx(7)-1);
    rttm.spkr_name{end+1}   = currLine(idx(7)+1:idx(8)-1);
    rttm.confidence{end+1}  = currLine(idx(8)+1:end);
end

fclose(FID);