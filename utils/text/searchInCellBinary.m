% THis function will search...
% INputs:
% lines: cell array of entries to be searched on
% pattern: query
% start: 

function found = searchInCellBinary(lines, pattern, start, stop)

found = 0;

if stop-start <= 10
    for i=start:stop
        if strcmp(lines{i}, pattern)
            found = i;
            return;
        end
    end

else
    
    middle = floor( (start+stop)/2 );
    notEqual = 0;
    middleWord = lines{middle};
    min_length = min( length(pattern), length(middleWord) );
    for i=1:min_length
        if int16(pattern(i)) > int16(middleWord(i))
            notEqual = 1;
            found = searchInCellBinary(lines, pattern, middle, stop);
            break;
        elseif int16(pattern(i)) < int16(middleWord(i))
            notEqual = 1;
            found = searchInCellBinary(lines, pattern, start, middle);
            break;
        end
    end
    
    if notEqual == 0
        if length(pattern) == length(middleWord)
            found = middle;
        elseif length(pattern) > length(middleWord)
            found = searchInCellBinary(lines, pattern, middle+1, stop);
        elseif length(pattern) < length(middleWord)
            found = searchInCellBinary(lines, pattern, start, middle-1);
        end
    end
end