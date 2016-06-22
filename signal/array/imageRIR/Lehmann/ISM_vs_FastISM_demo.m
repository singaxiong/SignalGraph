function [] = ISM_vs_FastISM_demo()
%ISM_vs_FastISM_demo  Compares results from ISM and fast-ISM simulations
%
% This Matlab function displays some typical results from the fast ISM
% implementation ('fast_ISM_RoomResp.m') and compares it with the same
% results obtained using a standard ISM simulation ('ISM_RoomResp.m').
%
% The function first generates a random simulation setup. It then 
% simulates a RIR according to the standard ISM implementation 
% ('ISM_RoomResp.m') together with a RIR according to the fast-ISM 
% implementation ('fast_ISM_RoomResp.m') for the same setup. The two 
% RIRs are then plotted together with their EDCs for comparison purposes.


% Simulation parameters:
Fs = 8000;              % sampling frequency in Hz
Delta_dB = 45;          % in dB, determines the length of the resulting RIRs
                        % (RIRs simulated until energy decays by Delta_dB).
Diffuse_dB = rand*12+8; % determines transition point from early reflections 
                        % to diffuse field in the fast ISM simulation. Here,
                        % pick a random value between 8 and 20 dB.
T60 = rand*0.4+0.2;     % reverberation time, random between .2 and .6 s


% The following code picks a random room setup with room volume between 20 
% and 250m^3, and with random absorption coefficient (uniform or fully random). 
% It also ensures that the environment is suitable for simulation with the
% function 'fast_ISM_RoomResp.m' (see function's help for more info).
okflag = 0;
while okflag==0
    roomvol = rand*230+20;      % select random room volume
    roomz = 0;
    while roomz<2 || roomz>5,   % room height between 2 and 5 meters
        roomx = rand*6+2;       % random room width between 2 and 6 meters
        roomy = rand*6+2;       % random room length between 2 and 6 meters
        roomz = roomvol/roomx/roomy;
    end
    room = [roomx roomy roomz];
    weights = ( rand(1,6)*.9+.1 ).^rand;    % uniform or random weights
	% determine corresponding absorption coefficients:
    [alpha,okflag] = ISM_AbsCoeff('t60',T60,room,weights,'LehmannJohansson'); 
	
    if okflag==1, 
		% Ensure that the environment is suitable for fast-ISM simulation
        % NOTE: the function 'fast_ISM_RoomResp.m' used further below also 
        %       performs the same check automatically and will display a 
        %       warning if the environment is unsuitable for simulation.
        Sx = room(2)*room(3); Sy = room(1)*room(3); Sz = room(1)*room(2);
        Avec = [Sx*alpha(1) Sx*alpha(2) Sy*alpha(3) Sy*alpha(4) Sz*alpha(5) Sz*alpha(6)];
        if std(Avec/(2*(Sx+Sy+Sz)))>0.035,
            okflag = 0;     % pick a different environment if current one unsuitable
        end
    end
end
beta = sqrt(1-alpha);   % reflection coefficients for current environment


% Pick a random position in the selected environment
X_src = rand(1,3).*room*.8 + .1*room;   % source position (avoid positions close to walls)
X_rcv = rand(1,3).*room*.8 + .1*room;   % mic position (avoid positions close to walls)
while norm(X_src-X_rcv)<.75,    % choose new points if they are too close to each other
    X_src = rand(1,3).*room*.8 + .1*room;
    X_rcv = rand(1,3).*room*.8 + .1*room;
end


% Print some details on screen:
fprintf('\n Details of the selected environment:\n');
fprintf('   room = [%.2f  %.2f  %.2f] (m)  -->  volume = %.1fm^3\n',room(1),room(2),room(3),prod(room));
fprintf('   source pos. = [%.2f  %.2f  %.2f] (m)\n',X_src(1),X_src(2),X_src(3));
fprintf('   sensor pos. = [%.2f  %.2f  %.2f] (m)\n',X_rcv(1),X_rcv(2),X_rcv(3));
fprintf('   absorption weights = [%.2f  %.2f  %.2f  %.2f  %.2f  %.2f]\n',weights(1)/max(weights),weights(2)/max(weights),weights(3)/max(weights),weights(4)/max(weights),weights(5)/max(weights),weights(6)/max(weights));
fprintf('   reflection coeffs = [%.2f  %.2f  %.2f  %.2f  %.2f  %.2f]\n',beta(1),beta(2),beta(3),beta(4),beta(5),beta(6));
fprintf('   T60 = %.3fs\n',T60);
fprintf('   Diffuse_dB = %.2fdB\n\n',Diffuse_dB);


% Compute the standard RIR (full-length)
stdRIR = ISM_RoomResp(Fs,beta,'t60',T60,X_src,X_rcv,room,'Delta_dB',Delta_dB);


% Compute the fast RIR
fprintf(' [fast_ISM_RoomResp] Computing transfer function: ');
fastRIR = fast_ISM_RoomResp(Fs,beta,'t60',T60,X_src,X_rcv,room,'Diffuse_dB',Diffuse_dB,'Delta_dB',Delta_dB);
fprintf('done!\n');


% Compute the EDCs for both RIRs (using Schroeder's method)
stdRIR_len = length(stdRIR);
stdRIR_tvec = [0:stdRIR_len-1]/Fs;
fastRIR_len = length(fastRIR);
fastRIR_tvec = [0:fastRIR_len-1]/Fs;

stdEDC_vec = zeros(1,stdRIR_len);
for nn=1:stdRIR_len,
    stdEDC_vec(nn) = sum(stdRIR(nn:end).^2);		% Energy decay using Schroeder's integration method
end
stdEDC_vec = 10*log10(stdEDC_vec/stdEDC_vec(1));	% Decay curve in dB.

fastEDC_vec = zeros(1,fastRIR_len);
for nn=1:fastRIR_len,
    fastEDC_vec(nn) = sum(fastRIR(nn:end).^2);      % Energy decay using Schroeder's integration method
end
fastEDC_vec = 10*log10(fastEDC_vec/fastEDC_vec(1));	% Decay curve in dB.


% Plot results
figure; subplot(3,1,1);
plot(stdRIR_tvec,stdRIR,'r');
axis tight; 
xlabel('time (s)'); ylabel('RIR');
title('standard ISM RIR');

subplot(3,1,2);
plot(fastRIR_tvec,fastRIR,'b');
axis tight;
xlabel('time (s)'); ylabel('RIR');
title('fast ISM RIR');

subplot(3,1,3);
plot(stdRIR_tvec,stdEDC_vec,'r'); hold on;
plot(fastRIR_tvec,fastEDC_vec,'b');
axis tight;
xlabel('time (s)'); ylabel('EDC (dB)');
title('EDCs of standard and fast RIRs')
