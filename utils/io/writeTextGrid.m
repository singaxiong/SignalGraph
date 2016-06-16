function writeTextGrid(fileName, convID, segment)


lines{1} = 'File type = "ooTextFile"';
lines{end+1} = 'Object class = "TextGrid"';
lines{end+1} = '';
lines{end+1} = sprintf('xmin = %f', segment.time1(1));
lines{end+1} = sprintf('xmax = %f', segment.time2(end));
lines{end+1} = 'tiers? <exists>';
lines{end+1} = 'size = 1 ';
lines{end+1} = 'item []: ';
lines{end+1} = '    item [1]:';
lines{end+1} = '        class = "IntervalTier" ';
lines{end+1} = sprintf('        name = "%s" ', convID);
lines{end+1} = sprintf('        xmin = %f', segment.time1(1));
lines{end+1} = sprintf('        xmax = %f', segment.time2(end));
lines{end+1} = sprintf('        intervals: size = %d', length(segment.time1)); 

for i=1:length(segment.time1)
    lines{end+1} = sprintf('        intervals [%d]:', i);
    lines{end+1} = sprintf('            xmin = %f', segment.time1(i));
    lines{end+1} = sprintf('            xmax = %f', segment.time2(i));
    lines{end+1} = sprintf('            text = "%s"', segment.label{i});
end

my_dump(fileName, lines);