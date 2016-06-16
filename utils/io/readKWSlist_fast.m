function kwlist = readKWSlist_fast(filename)

FID = fopen(filename);
fgetl(FID);

kwid = {};
oov_count = [];
search_time = [];
fileID = {};
channel = {};
tbeg = {};
dur = {};
score = {};
decision = {};
threshold = {};
raw_score = {};
verbose = 0;

currLine = fgetl(FID);
while 1
    if currLine == -1 
        break; 
    elseif length(regexp(currLine, '</kwslist>'))>0
        break;
    end
    
    idx = find(currLine=='"');
    search_time(end+1) = str2num(currLine(idx(1)+1:idx(2)-1));
    kwid{end+1} = currLine(idx(3)+1:idx(4)-1);
    if mod(length(kwid),300)==0
        fprintf('Now reading %dth keyword: %s - %s\n', length(kwid), kwid{end}, datestr(now));
    end
    oov_count(end+1) = str2num(currLine(idx(5)+1:idx(6)-1));
    fileID_tmp = {};
    channel_tmp = {};
    tbeg_tmp = {};
    dur_tmp = {};
    score_tmp = {};
    decision_tmp = [];
    threshold_tmp = [];
    raw_score_tmp = [];
    
    currLine = fgetl(FID);
    while 1
        if strfind(currLine, '</detected_kwlist>')
            break;
        end
        % idx = regexp(currLine, '"');
        idx = find(currLine=='"');  % this version is faster
        fileID_tmp{end+1} = currLine(idx(1)+1:idx(2)-1);
        channel_tmp{end+1} = currLine(idx(3)+1:idx(4)-1);
        tbeg_tmp{end+1} = currLine(idx(5)+1:idx(6)-1);
        dur_tmp{end+1} = currLine(idx(7)+1:idx(8)-1);
        score_tmp{end+1} = currLine(idx(9)+1:idx(10)-1);
        if strcmpi(currLine(idx(11)+1:idx(12)-1), 'YES')
            decision_tmp(end+1) = 1;
        else
            decision_tmp(end+1) = 0;
        end
        if length(idx)==16
            verbose = 1;
            threshold_tmp{end+1} = currLine(idx(13)+1:idx(14)-1);
            raw_score_tmp{end+1} = currLine(idx(15)+1:idx(16)-1);
        end
        
        currLine = fgets(FID);
    end
    
    fileID{end+1} = fileID_tmp;
    channel{end+1} = (channel_tmp);
    tbeg{end+1} = (tbeg_tmp);
    dur{end+1} = (dur_tmp);
    score{end+1} = (score_tmp);
    decision{end+1} = decision_tmp;
    if verbose
        threshold{end+1} = threshold_tmp;
        raw_score{end+1} = raw_score_tmp;
    end
    
    currLine = fgetl(FID);
end

kwlist.verbose = verbose;
kwlist.kwid = kwid;
kwlist.oov_count = oov_count;
kwlist.search_time = search_time;
kwlist.fileID = fileID;
kwlist.channel = channel;
kwlist.tbeg = tbeg;
kwlist.dur = dur;
kwlist.score = score;
kwlist.decision = decision;
if verbose
    kwlist.threshold = threshold;
    kwlist.raw_score = raw_score;
end

fclose(FID);
end


%%
function content = extractContent(field)
idx = regexp(field, '"');
content = field(idx(1)+1:idx(2)-1);
end
