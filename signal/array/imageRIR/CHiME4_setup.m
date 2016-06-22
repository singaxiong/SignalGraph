% This function set the parameters for simulating RIRs for CHiME3/4
% microphone array. 
% See Lehmann/ISM_setup.m for how to set the parameters for image method.
% Chenglin Xu / Xiong Xiao, 22 Jun 2016
%
function [SetupStruc] = CHiME4_setup(t60, room)
SetupStruc.Fs = 16000;                 % sampling frequency in Hz
SetupStruc.c = 343;                   % (optional) propagation speed of acoustic waves in m/s
micNumber = 6;      
SetupStruc.T60 = t60;                 % reverberation time T60, or define a T20 field instead!

% we define 3 room sizes
switch upper(room)
    case 'SMALL'
        SetupStruc.room = [6.00 4.00 3.00];           % room dimensions in m
    case'MEDIUM'
        SetupStruc.room = [10.00 8.00 3.00];           % room dimensions in m    
    case 'LARGE'
        SetupStruc.room = [14.00 12.00 3.00];       % original [15,10,3]   % room dimensions in m    
    otherwise error('Room size is not defined...');
end;
if isequal(t60,0)
    beta = 0;
else
    beta = exp(-13.82/sum(1./SetupStruc.room(1,:)*SetupStruc.c*t60));%calculate the reflection coefficient
end;
SetupStruc.reflect_weights = ones(1,micNumber)*beta;    % (optional) weights for the resulting alpha coefficients.

% Define microphone positions in meters
xmic=[-0.10 0 0.10 -0.10 0 0.10]'; % left to right axis
ymic=[0.095 0.095 0.095 -0.095 -0.095 -0.095]'; % bottom to top axis
zmic=[0 -0.02 0 0 0 0]'; % back to front axis
micCenter = SetupStruc.room./2;
SetupStruc.mic_pos = repmat(micCenter,micNumber,1) + [xmic ymic zmic];

%% Uncomment the following for a 3D plot of the above setup:
% plot3(SetupStruc.src_traj(:,1),SetupStruc.src_traj(:,2),SetupStruc.src_traj(:,3),'ro-','markersize',4); hold on;
% plot3(SetupStruc.mic_pos(:,1),SetupStruc.mic_pos(:,2),SetupStruc.mic_pos(:,3),'ko','markerfacecolor',ones(1,3)*.6);
% axis equal; axis([0 SetupStruc.room(1) 0 SetupStruc.room(2) 0 SetupStruc.room(3)]);
% box on; xlabel('x-axis (m)'); ylabel('y-axis (m)'); zlabel('z-axis (m)');
