% check whether the sequences are of the same length. 
% The sequences are aligned from begining. If some sequence is shorter, it
% will be padded with a big negative number, i.e. -1e10, in the first dimension.  
%
function [mask, variableLength] = CheckTrajectoryLength(data)

mask = squeeze(data(1,:,:)) == -1e10;
if sum(sum(mask))==0    % all have equal length
    variableLength = 0;
else
    variableLength = 1;
end
end