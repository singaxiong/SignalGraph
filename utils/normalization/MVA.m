function output = MVA(input,M)

output = input;
% M is the order of the filter
for j=M+1:size(output,1)-M
    output(j,:) = ( sum(output(j-M:j-1,:),1) + sum(input(j:j+M,:)) )/(2*M+1);
end
