function [FigOH,Exists] = reusefig(TagName,varargin)
%reusefig  Creates and/or re-uses a figure with specific tag
%
% [FigOH,Exists] = reusefig(TagName,'PropName1',PropVal1,'PropName2',PropVal2,...)
%
% Looks for the figure with tag 'TagName' and makes it current, or 
% creates it if necessary. Returns the handle 'FigOH' of the figure.
%
% This function can be used in order to re-use the same plot windows
% for several calls of the same plotting routine (avoids creating new
% figures at every call or plotting in windows "belonging" to other 
% functions).
%
% A list of arguments pairs can be passed on to this function which will 
% use them to set the various properties of the figure (see SET).
%
% The function returns Exists=1 if the figure already existed before, 0 if 
% it had to create a new figure.

% Release date: August 2008
% Author: Eric A. Lehmann, Perth, Australia (www.eric-lehmann.com)
%
% Copyright (C) 2008 Eric A. Lehmann
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.

FigOH = findobj('type','figure','tag',TagName);

if isempty(FigOH),
    % Create a new figure if no exisiting one found:
    FigOH = figure('tag',TagName,'IntegerHandle','off','NumberTitle','off','Name',TagName,varargin{:});  
    % Uses non-integer handle to reduce the probability of other functions inadvertently
    % plotting into it (which is why the function reusefig is done for!).
    Exists = 0;
elseif isequal(size(FigOH),[1 1]),
    figure(FigOH);              % If already existing, makes it current and brings it to screen.
    if ~isempty(varargin),
       set(FigOH,varargin{:});	% Set figure properties to desired user values.
    end
    Exists = 1;
    %shg;
else
    error('More than one object found with same tag!');
end
