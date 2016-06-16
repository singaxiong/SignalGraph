function [vad_flag, energyVAD_flag, periodicVal] = CombVAD(wav, Fs, DEBUG)
%inputList = '../VAD/short1_logmmse.wav';
Gain = 1;
[Test.wave] = wav;
[Ref.wave] = wav;
Test.wave = Gain.*Test.wave;

max_nSampleTotal = length(Test.wave);
FrameSzTime     = 0.1;  % ?? msec frame Window
% the current wisdom is to use bin Bandwidth of 10 Hz for FFT resolution
FrameForward    = 0.02;  % ?? No overlap in this example

nSamplePerFrame = FrameSzTime * Fs;
nSampleForward  = FrameForward * Fs;

numFrame  = floor((max_nSampleTotal - nSamplePerFrame)/nSampleForward);

s_new.frameIdx = 0;
s_new.MinFeatureVal = 15;
s_new.MinRatioSpeechPeakNoiseFloor = 2.5;
s_new.ThOffset  = 10;

Test.periodicVADVal = batch_peridoc_pitch_count_fast(Test.wave, Fs, numFrame, nSamplePerFrame, nSampleForward);

x_start   = 1;
x_end     = nSamplePerFrame;
for j=1:numFrame
    
    Test.frameID = j;
    Test.frame_x =  Test.wave(x_start:x_end);
    Ref.frame_x  =  Ref.wave(x_start:x_end);
    
    Ref.energyVADVal(j)    = energyVADVal(Ref.frame_x, Fs);
    
    s_new = adaptThreshold(s_new,Test.periodicVADVal(j));
    
    Test.th(j)       = s_new.th;
    Test.th2(j)      = s_new.th2;
    Test.speecVar(j) = s_new.speechVar;
    Test.nf(j)       = s_new.nf;
    Test.slow_nf(j)  = s_new.slow_nf;
    Test.vadFlag(j)  = s_new.vadFlag;
    
    x_start = x_start + nSampleForward;
    x_end   = x_end   + nSampleForward;
    
    if (mod(j,10) == -1)
        figure(3);
        plot(Test.periodicVADVal,'b'); hold on;
        plot(Test.th2,'r');
        plot(-100.*Test.vadFlag,'g');
        plot(-10.*Ref.energyVADVal,'k'); hold on;
        pause(0.01);
    end
end
if DEBUG
    plot(Test.periodicVADVal,'b'); hold on;
    plot(Test.th2,'r');
    plot(-100.*Test.vadFlag,'g');
    plot(-10.*Ref.energyVADVal,'k'); hold off;
    pause(0.1);
end

vad_flag = Test.vadFlag;
energyVAD_flag = Ref.energyVADVal;
periodicVal = Test.periodicVADVal;
end
