
function phase = get_principal_value(phase)
idx = find(phase>pi);
phase(idx) = phase(idx) - 2*pi;

idx = find(phase<-pi);
phase(idx) = phase(idx) + 2*pi;

end
