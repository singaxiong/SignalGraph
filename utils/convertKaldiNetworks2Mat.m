function convertKaldiNetworks2Mat(kaldiFileName, matFileName)

fprintf('Loading Kaldi network from %s!\n', kaldiFileName);
kaldiNetwork = readKaldiNetwork(kaldiFileName);

if nargin<2
    matFileName = [kaldiFileName '.mat'];
end
save(matFileName, 'kaldiNetwork');
fprintf('Saved Kaldi network into %s!\n', matFileName);
