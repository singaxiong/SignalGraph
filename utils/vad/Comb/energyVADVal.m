function  [energyVal] =  energyVADVal(frame_x, Fs)
energyVal = norm(frame_x);