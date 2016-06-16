function data = read_line_by_line(FID);

% first pass, find the number of lines
cnt = 1;
while 1
    tmp = fgetl(FID);
    if tmp == -1
        break;
    else
        cnt = cnt + 1;
    end
end

% second pass, read the data
frewind(FID);
data = cell(cnt-1,1);
cnt = 1;
while 1
    tmp = fgetl(FID);
    if tmp == -1
        break;
    else
        data{cnt} = tmp;
        cnt = cnt + 1;
    end
end

