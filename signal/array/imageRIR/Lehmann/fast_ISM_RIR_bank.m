function [RIR_cell] = fast_ISM_RIR_bank(setupstruc,RIRFileName,varargin)
%fast_ISM_RIR_bank  Bank of RIRs using Lehmann & Johansson's fast image-source method
%
% [RIR_CELL] = fast_ISM_RIR_bank(SETUP_STRUC,RIR_FILE_NAME)
% [RIR_CELL] = fast_ISM_RIR_bank( ... ,'arg1',val1,'arg2',val2,...)
%
% This function generates a bank of room impulse responses (RIRs) for a
% particular user-defined room setup, using Lehmann and Johansson's fast
% implementation of the image-source method (diffuse reverberation model,
% see: "Diffuse reverberation model for efficient image-source simulation 
% of room impulse responses", IEEE Trans. Audio, Speech, Lang. Process., 
% 2010). The input SETUP_STRUC is a structure of enviromental parameters 
% containing the following fields:
%
%          Fs: sampling frequency (in Hz).
%        room: 1-by-3 vector of enclosure dimensions (in m), 
%              [x_length y_length z_length].
%     mic_pos: N-by-3 matrix, [x1 y1 z1; x2 y2 z2; ...] positions of N
%              microphones (in m). 
%    src_traj: M-by-3 matrix, [x1 y1 z1; x2 y2 z2; ...] positions of M 
%              source trajectory points (in m).
%  T20 or T60: scalar value (in s), desired reverberation time.
%           c: (optional) sound velocity (in m/s).
% abs_weights: (optional) 1-by-6 vector of absorption coefficients weights, 
%              [w_x1 w_x2 w_y1 w_y2 w_z1 w_z2].
%
% If the field SETUP_STRUC.c is undefined, the function assumes a default
% value of sound velocity of 343 m/s.
%
% The field 'abs_weight' corresponds to the relative weights of each of the
% six absorption coefficients resulting from the desired reverberation time.
% For instance, defining 'abs_weights' as [1 1 0.8 0.8 0.6 0.6] will result
% in the absorption coefficients (alpha) for the walls in the y-dimension
% being 20% smaller compared to the x-dimension walls, whereas the floor
% and ceiling will end up with absorption coefficients 40% smaller (e.g.,
% to simulate the effects of a concrete floor and ceiling). If this field
% is omitted, the parameter 'abs_weight' will default to [1 1 1 1 1 1],
% which leads to uniform absorption coefficients for all room boundaries.
%
% The structure SETUP_STRUC may contain one of the two fields 'T60' or
% 'T20'. This function will automatically determine which reverberation
% type is used and compute the desired room absorption coefficients
% accordingly. T20 is defined as the time required for the impulse response
% energy to decay from -5 to -25dB, whereas T60 corresponds to the time
% required by the impulse response energy to decay by 60dB. Setting the 
% corresponding field value to 0 achieves anechoic impulse responses 
% (direct path only).
%
% In addition, a number of other (optional) parameters can be set using a 
% series of 'argument'--value pairs. The following parameters (arguments)
% can be used:
%
%   'Delta_dB': scalar (in dB), parameter determining how much the resulting 
%               impulse response is cropped: the impulse response is
%               computed until the time index where its overall energy
%               content has decreased by 'Delta_dB' decibels, after which 
%               the computations stop. Not relevant if the reverberation 
%               time is set to 0 (anechoic case). Defaults to 50.
% 'Diffuse_dB': scalar (in dB), determines the desired transition point
%               from early reflections to diffuse field in the fast ISM
%               simulations: the diffuse reflections are assumed to start
%               from the time index where the RIR's energy has decayed by
%               'Diffuse_dB' decibels, following which the diffuse field
%               model is applied (see above mentioned reference). Defaults
%               to 13.
% 'SilentFlag': set to 1 to disable this function's on-screen messages. 
%               Defaults to 0.
%
% This function returns a 2-dimensional cell array RIR_CELL containing the
% RIRs for each source trajectory point and each microphone, organised as
% follows: RIR_CELL{mic_index,traj_index}. The resulting filter length
% may differ slightly in each computed RIR.
%
% This function also saves the computation results on file. The argument
% RIR_FILE_NAME determines the name of the .mat file where the variable
% RIR_CELL is to be saved. If a file already exists with the same name as
% the input argument, the user will be prompted to determine whether the
% file is to be overwritten or not. The given parameter RIR_FILE_NAME can
% be a full access path to the desired file. If no access path is given,
% the file is saved in the current working directory. 

% Release date: November 2009
% Author: Eric A. Lehmann, Perth, Australia (www.eric-lehmann.com)
%
% Copyright (C) 2009 Eric A. Lehmann
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

VarList = {'Delta_dB'       50;         % maximum attenuation in RIR computation
           'Diffuse_dB'     13;         % diffuse field limit
           'SilentFlag'     0};         % set to 1 to disable on-screen messages
eval(SetUserVars(VarList,varargin));    % set user-definable variables

if length(RIRFileName)<=4 || ~strcmpi(RIRFileName(end-3:end),'.mat'),
    RIRFileName = [RIRFileName '.mat'];
end

if exist(RIRFileName,'file')==2,
    foo = input(' [fast_ISM_RIR_bank] The file name passed as input already exists. Overwrite? [yes/no]: ','s');
    if ~strcmpi(foo,'yes');
        fprintf(' [fast_ISM_RIR_bank] Terminating execution now (no data was saved).\n');
        return
    end
end

Fs = setupstruc.Fs;
room = setupstruc.room;
micpos = setupstruc.mic_pos;
straj = setupstruc.src_traj;

if isfield(setupstruc,'abs_weights'),
    weights = setupstruc.abs_weights;
else
    weights = ones(1,6);
end
if isfield(setupstruc,'c'),
    cc = setupstruc.c;
else
    cc = 343;
end
if isfield(setupstruc,'T60'),
    alpha = ISM_AbsCoeff('t60',setupstruc.T60,room,weights,'LehmannJohansson','c',cc);
    rttype = 'T60'; rtval = setupstruc.T60;
elseif isfield(setupstruc,'T20'),
    alpha = ISM_AbsCoeff('t20',setupstruc.T20,room,weights,'LehmannJohansson','c',cc);
    rttype = 'T20'; rtval = setupstruc.T20;
else
    error('Missing T60 or T20 field.');
end
beta = sqrt(1-alpha);

nMics = size(micpos,1);     % number of microphones
nSPts = size(straj,1);      % number of source trajectory points

%-=:=- Check for irregular decay pattern -=:=-
Sx = room(2)*room(3); Sy = room(1)*room(3); Sz = room(1)*room(2);
Avec = [Sx*alpha(1) Sx*alpha(2) Sy*alpha(3) Sy*alpha(4) Sz*alpha(5) Sz*alpha(6)];
irreg_crit = std(Avec/(2*(Sx+Sy+Sz)));
if irreg_crit>0.035,%%0.04,
    fprintf(' [fast_ISM_RIR_bank] Warning! The uneven distribution of sound absorption in the considered \n');
    fprintf('   environment might lead to an irregular decay pattern in the computed RIRs, which cannot be \n');
    fprintf('   modeled properly with the current approximation method. It is suggested that you select an \n');
    fprintf('   environment with more uniform values of absorption coefficients instead, or use a full ISM \n');
    fprintf('   simulation (''ISM_RIR_bank.m'') in order to achieve proper impulse responses. \n');
    foo = input(' [fast_ISM_RIR_bank] Do you still want to proceed with the current simulation? [yes/no]: ','s');
    if ~strcmpi(foo,'yes'),
        fprintf(' [fast_ISM_RIR_bank] Terminating execution now (no data was saved).\n');
        return
    end
end

%-=:=- Compute RIR bank -=:=-
RIR_cell = cell(nMics,nSPts); % pre-allocate cell array
if ~SilentFlag, PrintLoopPCw(' [fast_ISM_RIR_bank] Computing room impulse responses. '); end;
for mm=1:nMics,
    X_rcv = micpos(mm,:);
    for tt=1:nSPts,         % compute ISM room impulse response for each source-receiver combinations
        if ~SilentFlag, PrintLoopPCw((mm-1)*nSPts+tt,nMics*nSPts); end;
        X_src = straj(tt,:);
        RIR_cell{mm,tt} = fast_ISM_RoomResp(Fs,beta,rttype,rtval,X_src,X_rcv,room,'c',cc, ...
            'Delta_dB',Delta_dB,'Diffuse_dB',Diffuse_dB,'DisableIrregDecayWarnings',1);
    end
end

%-=:=- Save results into .mat file -=:=-
save(RIRFileName,'RIR_cell');
if ~SilentFlag, fprintf(' [fast_ISM_RIR_bank] RIR bank parameter ''RIR_cell'' saved in file ''%s''\n',RIRFileName); end;
