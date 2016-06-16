function y = rastafilt(x,parameter)
% y = rastafilt(x)
%
% rows of x = critical bands, cols of x = frame
% same for y but after filtering
% 
% default filter is single pole at 0.94

if nargin<2
    parameter = -.94;
end

% rasta filter
numer = [-2:2];
numer = -numer ./ sum(numer.*numer);

denom = [1 parameter];

% Initialize the state.  This avoids a big spike at the beginning 
% resulting from the dc offset level in each band.
% (this is effectively what rasta/rasta_filt.c does).
% Because Matlab uses a DF2Trans implementation, we have to 
% specify the FIR part to get the state right (but not the IIR part)
[y,z] = filter(numer, 1, x(:,1:4)');
% .. but don't keep any of these values, just output zero at the beginning
y = 0*y';

% Apply the full filter to the rest of the signal, append it
y = [y,filter(numer, denom, x(:,5:end)',z)'];

