% WSJCAM0 audio files are in SPHERE format, while the rest of REVERB
% Challenge data are in wav format. To make it easier to read audio files,
% we first save the WSJCAM0 audio into wav files. 
% Xiong Xiao, Nanyang Technological University, Singapore. 
% Feb 9, 2017
%
function ResaveWSJCAM0WAV()

wsjcam0root = 'D:/Data/wsjcam0';
sph_root = [wsjcam0root '/data'];
wav_root = [wsjcam0root '/data_wav'];

% for our purpose, it is enough to just use wv1 files. 
files = findFiles(sph_root, 'wv1');

for i=1:length(files)
    wav = readsph(files{i}); % read SPHERE files by using function from voicebox.
    output_file = [wav_root files{i}(length(sph_root)+1:end-3) 'wav'];
    filepath = fileparts(output_file);
    my_mkdir(filepath);
    audiowrite(output_file, wav, 16000);
    PrintProgress(i, length(files), 100, 'Convert SPHERE file into WAV file');
end

end
