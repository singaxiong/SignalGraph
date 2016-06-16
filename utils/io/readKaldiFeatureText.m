function [name, feat] = readKaldiFeatureText(fileName)

FID = fopen(fileName, 'r');
name = {};
feat = {};
while 1
    line = fgetl(FID);
    if line==-1
        break;
    end
    
    idx = regexp(line, '[');
    if length(idx)==0 
        break;
    end
    
    idx = regexp(line, ' ');
    name{end+1} = line(1:idx(1)-1);

    frames = {};
    while 1
        frames{end+1} = fgetl(FID);
        idx = regexp(frames{end}, ']');
        if length(idx)>0
            frames{end} = frames{end}(1:idx-1);
            break;
        end
    end
    
    % convert the frames from string to numbers
    dim = length(str2num(frames{1}));
    feat{end+1} = zeros(dim, length(frames));
    
    for i=1:length(frames)
        feat{end}(:,i) = str2num(frames{i});
    end
end
fclose(FID);

