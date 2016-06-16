% Decide which the alphabetic order of two strings. 
%
% y = 1 if str1 is in front of str2
%     0 if two strings are identical
%     -1 if str2 is behind str2

function y = strcmp_alpha_order(str1, str2)

len1 = length(str1);
len2 = length(str2);

len = min(len1, len2);

y = 0;
for i=1:len
    val1 = int16(str1(i));
    val2 = int16(str2(i));
    if val1 > val2
        y = 1;
        break;
    elseif val1 < val2
        y = -1;
        break;
    end    
end

if y == 0
    if len1 > len2
        y = 1;
    elseif len1 < len2
        y = -1;
    end
end
    