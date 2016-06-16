% Convert an integer to string, but add optional 0's in the front to make
% the string equal to a predefined width. 
% E.g.
% if num=10, width=3, then str = '010';
% if num=10, width=2, then str = '10';
% if num=10, width=1, then str = '10';
% Author: Xiong Xiao, Temasek Lab @ NTU, Singapore. 
% Created: 09 Jun 2013
% Last Modified: 09 Jun 2013
function str = num2str_fixed_width(num, width)

str = num2str(num);
if length(str)>width
    fprintf('Warning: string width is larger than predefined value!\n');
elseif length(str)<width
    while 1
        str = ['0' str];
        if length(str)==width
            break;
        end
    end
end
