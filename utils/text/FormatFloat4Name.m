function s = FormatFloat4Name(x)

if abs(x) - floor(abs(x)) == 0  % check whether it is an integer
    s = num2str(x);
else
%     if abs(x)>=10 || abs(x) <= 0.1    % use scientific representation
        s = sprintf('%2.2E', x);
        idx = regexp(s, 'E');
        digit = str2num(s(1:idx-1));
        exp_term = str2num(s(idx+1:end));
        if exp_term == 0
            s = num2str(digit);
        else
            s = [num2str(digit) 'E' num2str(exp_term)];
        end
%         s = [RemoveEndingZero(s(1:idx-1)) s(idx) RemovePreceedingZero(s)];
%     else            % use floating point representation
%         s = sprintf('%2.2f', x);
%         s = RemoveEndingZero(s);
%     end
end
end


%% remove ending zero

function s = RemoveEndingZero(s)    % remove 0's at the end of floating points
for i = length(s):-1:1
    if s(end)== '0'
        s(end) = [];
    else
        break;
    end
end
if s(end)=='.'
    s(end) = [];
end
end

function s = RemovePreceedingZero(s)
remove_idx = [];
for i=1:length(s)
    if s(i) == '0'
        remove_idx(end+1) = i;
    else
        break;
    end
end
s(remove_idx) = [];
end
