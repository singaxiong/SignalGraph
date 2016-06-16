function [text, timeStamp] = ReadPraatTextGrid(transFile)
feature('DefaultCharacterSet', 'UTF8');

trans = my_cat(transFile);

for i=1:length(trans)
    if length(trans{i})>4 && strcmp(trans{i}(1:4), 'item')
        break;
    end
end
i = i+ 6;
idx = regexp(trans{i}, '=');
nInterval = str2num(trans{i}(idx+1:end));

i = i + 1;
for cnt = 1:nInterval
    idx = regexp(trans{i+1}, '=');
    timeStamp(1,cnt) = str2num(trans{i+1}(idx+1:end));
    
    idx = regexp(trans{i+2}, '=');
    timeStamp(2,cnt) = str2num(trans{i+2}(idx+1:end));

    idx = regexp(trans{i+3}, '"');
    text{cnt} = trans{i+3}(idx(1)+1:idx(end)-1);
    i = i+4;
    
%      fprintf('%2.2f %2.2f %s\n', timeStamp(1,cnt), timeStamp(2,cnt), text{cnt});
end

cnt = 1;
text2 = {};
for j=1:length(text)
    if length(text{j})>0
        text2{cnt} = text{j};
        timeStamp2(:,cnt) = timeStamp(:,j);
        cnt = cnt + 1;
    end
end

text = text2;
timeStamp = timeStamp2;