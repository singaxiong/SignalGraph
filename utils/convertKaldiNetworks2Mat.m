function convertKaldiNetworks2Mat(kaldiFileName, matFileName, precision)
if nargin<3
    precision = 'single';
end

fprintf('Loading Kaldi network from %s!\n', kaldiFileName);
kaldiNetwork = readKaldiNetwork(kaldiFileName);

kaldiNetwork = convertKaldiNetworksPrecision(kaldiNetwork, precision);

if nargin<2
    matFileName = [kaldiFileName '.mat'];
end

save(matFileName, 'kaldiNetwork');
fprintf('Saved Kaldi network into %s!\n', matFileName);
