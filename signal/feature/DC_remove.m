% dc_remove remove the DC offset of the time domain signal

function [z] = DC_remove(x,a)

Num = [a -a];
Dem = [1 -a];
z = filter(Num,Dem,x);

