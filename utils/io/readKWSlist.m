function kwlist = readKWSlist(filename)

lines = my_cat(filename);

tmp = ExtractWordsFromString_v2(lines{1});

kwid = {};
oov_count = [];
search_time = [];
fileID = {};
channel = {};
tbeg = {};
dur = {};
score = {};
decision = {};

i=2;
while 1
    if i>length(lines) || length(regexp(lines{i}, '</kwslist>'))>0
        break; 
    end
    
    tmp = ExtractWordsFromString_v2(lines{i});
    search_time(end+1) = str2num(extractContent(tmp{2}));
    kwid{end+1} = extractContent(tmp{3});
    if mod(length(kwid),100)==0
        fprintf('Now reading %dth keyword: %s - %s\n', length(kwid), kwid{end}, datestr(now));
    end
    oov_count(end+1) = str2num(extractContent(tmp{4}));
    fileID{end+1} = {};
    channel{end+1} = [];
    tbeg{end+1} = [];
    dur{end+1} = [];
    score{end+1} = [];
    decision{end+1} = [];
    
    i=i+1;
    while 1
        if length(regexp(lines{i}, '</detected_kwlist>'))>0
            break;
        end
        tmp = ExtractWordsFromString_v2(lines{i});
        fileID{end}{end+1} = extractContent(tmp{2});
        channel{end}(end+1) = str2num(extractContent(tmp{3}));
        tbeg{end}(end+1) = str2num(extractContent(tmp{4}));
        dur{end}(end+1) = str2num(extractContent(tmp{5}));
        score{end}(end+1) = str2num(extractContent(tmp{6}));
        if strcmpi(extractContent(tmp{7}), 'YES')
            decision{end}(end+1) = 1;
        else
            decision{end}(end+1) = 0;
        end
        i = i+1;
    end
    i = i+1;
end

kwlist.kwid = kwid;
kwlist.oov_count = oov_count;
kwlist.search_time = search_time;
kwlist.fileID = fileID;
kwlist.channel = channel;
kwlist.tbeg = tbeg;
kwlist.dur = dur;
kwlist.score = score;
kwlist.decision = decision;

end


%%
function content = extractContent(field)
idx = regexp(field, '"');
content = field(idx(1)+1:idx(2)-1);
end
