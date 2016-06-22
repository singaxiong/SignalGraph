function [EvalStr] = SetUserVars(VarList,varargin)
%SetUserVars  Checks and sets user-definable variables
%
% [evalstr] = SetUserVars(VarList,UserVarCell)
%
% This function reads in a series of 'ArgName' -- 'ArgVal' parameters from
% the cell array 'UserVarCell' and checks them with the list of
% user-definable parameters in the cell array 'VarList'. It returns a
% string 'evalstr' that can be evaluated back in the calling function and
% that sets the parameters to either the user-defined values 'ArgVal' or
% the default values in 'VarList'. Use as in example below.
%
%   function [] = DoIt(DoIt_in,varargin);
%   %DoIt  Executes some code
%   %[DoIt_out] = DoIt(DoIt_in,'ArgName1',ArgVal1,'ArgName2',ArgVal2,...)
%   %This function accepts a series of argument pairs to set user variables
% 
%   %User-definable parameters with default values:
%   VarList = {'DoItParam1'   5.7;         % numerical parameter
%              'DoItParam2'   [1 0; 0 1];  % matrix parameter
%              'DoItParam3'   'none'};     % string parameter
%   eval(SetUserVars(VarList,varargin));   % set the function parameters to either the 
%                                          % default values or user-defined values
% 
%    ... your "DoIt" function code using variables DoItParam1, DoItParam2, etc. ...
% 

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

if length(varargin)==1,
    varargin = varargin{1};     % if SetUserVars is called within other function as in the example above
end

if mod(length(varargin),2)~=0,
    error('Number of arguments is odd: arguments must be passed in pairs (''ArgName'',ArgVal).'); 
end

EvalStr = [];
NumSetVars = 0;
for ii=1:size(VarList,1),
   tmp = find(strcmp(varargin,VarList{ii,1}));		% Find argument index in varargin cell vector.
   if length(tmp)==1,	% If argument is found and unique, set variable appropriately.
       if ischar(varargin{tmp+1}),
           EvalStr = [EvalStr VarList{ii,1} '=''' varargin{tmp+1} ''';'];
       else
           EvalStr = [EvalStr VarList{ii,1} '=' mat2str(varargin{tmp+1}) ';'];
       end
       NumSetVars = NumSetVars + 1;			% Check the number of variables set from varargin.
   else                 % Assign default value if parameter not passed as function argument.
       if ischar(VarList{ii,2}),
           EvalStr = [EvalStr VarList{ii,1} '=''' VarList{ii,2} ''';'];
       else
           EvalStr = [EvalStr VarList{ii,1} '=' mat2str(VarList{ii,2}) ';'];
       end
   end
end

if NumSetVars~=length(varargin)/2,
   error('Some of the input arguments could not be set properly (check syntax)')
end
