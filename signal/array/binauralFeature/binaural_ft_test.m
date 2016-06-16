clear all;
echo off;
close all;

load sa2.mat;

soundVelocity = 343;      
micPos = [1.000 2.400
                 1.000 2.500];
micDist = norm(micPos(2,:)-micPos(1,:));
srcPos = [1.5 0.5];
% delays
t1 = norm(srcPos-micPos(1,:))/soundVelocity;
t2 = norm(srcPos-micPos(2,:))/soundVelocity;
delay = [t1, t2];
% decays
r1 = 1/(4*pi*norm(srcPos-micPos(1,:)));
r2 = 1/(4*pi*norm(srcPos-micPos(2,:)));
decay = [r1 r2];
% create delayed signals
y1 = [audioSignal audioSignal];%ones(microNumber,1)*resampleAudio';
[micSignalTemp,tn] = delaySignal(y1,fs,delay,length(audioSignal),'simpleDelay');
micSig = repmat(decay,length(tn),1).*micSignalTemp;
% extract binaural features
frameLength=512;
overlap=0.75;
[tauDiffFull, ampDiffFull, td, ld] = binaural_ft(micSig,fs,frameLength,overlap,micDist);
