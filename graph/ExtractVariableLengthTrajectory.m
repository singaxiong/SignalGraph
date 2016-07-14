% check whether the sequences are of the same length. 
% The sequences are aligned from begining. If some sequence is shorter, it
% will be padded with a big negative number, i.e. -1e10, in the first dimension.  
%
function [data2, mask, variableLength] = ExtractVariableLengthTrajectory(data, mask)
if nargin<2
    mask = squeeze(data(1,:,:))== -1e10;
end

[D,T,N] = size(data);
if T==1 || N==1
    mask = mask';
end
if sum(sum(mask))==0    % all have equal length
    variableLength = 0;
else
    variableLength = 1;
end
for i=1:N
    data2{i} = data(:,mask(:,i)==0,i);
end
end