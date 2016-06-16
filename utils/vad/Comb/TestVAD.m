function TestVAD(wavRoot, output, step, savestep, DEBUG)
if nargin<3
    step = 1;
end
if nargin<4
    savestep=50;
end
if nargin<5
    DEBUG = 1;
end

files = my_dir([wavRoot '\*.wav']);

for i=1:length(files)
    PrintProgress(i, length(files), step, files{i});
    [wav, Fs] = wavread([wavRoot '\' files{i}]);
    [comb_vad_flag{i}, energy_val{i}, periodicVal{i}] = CombVAD(wav, Fs, DEBUG);
    if mod(i,savestep)==0
        save(output, 'comb_vad_flag', 'energy_val', 'periodicVal', 'files');
    end
end
save(output, 'comb_vad_flag', 'energy_val', 'periodicVal', 'files');

% example of post processing

% double frame rate
currVAD = comb_vad_flag{1};
currVAD = interp1([1:length(currVAD)]*2, currVAD, 1:(2*length(currVAD)), 'nearest', 'extrap');

% Post process VAD with a buffer length of 0.4s
[currVAD_merged, currVAD_extended] = PostProcessVAD(currVAD, 40);

end