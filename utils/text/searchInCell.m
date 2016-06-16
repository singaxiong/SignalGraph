


function match = searchInCell(lines, pattern, type, start_point)

match = 0;

if nargin<3
    type = 'contain';
end
if nargin<4
    start_point = 1;
end

switch type
    case 'contain'
        tmp = strfind(lines,pattern);
        for i=1:length(lines)
            %     if length(regexp(lines{i},pattern))>0
            if tmp{i}>0
                match(i) = 1;
            else
                match(i) = 0;
            end
        end
        
    case 'equal'
        tmp = strfind(lines,pattern);
        for i=1:length(lines)
            if tmp{i}>0 && strcmp(lines{i},pattern)
                match(i) = 1;
            else
                match(i) = 0;
            end
        end
    case 'equal_found_and_leave'
        tmp = strfind(lines,pattern);
        for i=start_point:length(lines)
            if ~isempty(tmp{i}) && strcmp(lines{i},pattern)
                match(i) = 1;
                break;
            else
                match(i) = 0;
            end
        end
        
    case 'equal_found_and_leave_binary'
        searchInCellBinary(lines, pattern);
end

