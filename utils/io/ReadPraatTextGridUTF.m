function [text, timeStamp] = ReadPraatTextGridUTF(transFile)
% feature('DefaultCharacterSet', 'Shift_JIS');

trans = my_cat(transFile);

for i=1:length(trans)
    curr_trans = unicode2ascii(trans{i});
    if length(curr_trans)>4 && length(strfind(curr_trans, 'item'))>0
        break;
    end
end
i = i+ 12;
curr_trans = unicode2ascii(trans{i});
idx = regexp(curr_trans, '=');
nInterval = str2num(curr_trans(idx+1:end));

i = i + 2;
for cnt = 1:nInterval
    curr_trans = unicode2ascii(trans{i+2});
    idx = regexp(curr_trans, '=');
    timeStamp(1,cnt) = str2num(curr_trans(idx+1:end));
    
    curr_trans = unicode2ascii(trans{i+4});
    idx = regexp(curr_trans, '=');
    timeStamp(2,cnt) = str2num(curr_trans(idx+1:end));
    
    curr_trans = (trans{i+6});
    idx = regexp(curr_trans, '"');
    if length(idx)>=2
        text{cnt} = curr_trans(idx(1)+1:idx(end)-1);
        i = i+8;
    else
        text{cnt} = curr_trans(idx(1)+1:end);
        curr_trans = (trans{i+7});
        idx = regexp(curr_trans, '"');
        text{cnt} = [text{cnt} curr_trans(1:idx(1)-1)];
        i = i+9;
    end
    
    %fprintf('%2.2f %2.2f %s\n', timeStamp(1,cnt), timeStamp(2,cnt), text{cnt});
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


%%

function output = unicode2ascii(input)

b = unicode2native(input);
output = native2unicode(b(2:2:end));
