function [scores] = RunObjectiveMeasures(cleanWav, enhancedWav, measures, fs, DEBUG)
addpath('../../../../Enhancement/Loizou/MATLAB_code/objective_measures/quality');

cleanWav = cleanWav(:);

for i = 1:length(measures)
    for j = 1:length(enhancedWav)
        currEnhancedWav = enhancedWav{j}(:);
        nSample = min(length(currEnhancedWav), length(cleanWav));
        switch lower(measures{i})
            case 'pesq'
                scores(i,j) = pesq_fast_vectorize(cleanWav(1:nSample), currEnhancedWav(1:nSample), 0, fs);
            case 'fwseq'
                scores(i,j) = comp_fwseg_fast(cleanWav(1:nSample), currEnhancedWav(1:nSample), 0, fs);
            case 'cd'
                scores(i,j) = comp_cep(cleanWav(1:nSample), currEnhancedWav(1:nSample), 0, fs);
            case 'is'
                scores(i,j) = comp_is(cleanWav(1:nSample), currEnhancedWav(1:nSample), 0, fs);
            case 'llr'
                scores(i,j) = comp_llr(cleanWav(1:nSample), currEnhancedWav(1:nSample), 0, fs);
            case 'wss'
                scores(i,j) = comp_wss(cleanWav(1:nSample), currEnhancedWav(1:nSample), 0, fs);
        end
    end
%     if DEBUG
        fprintf('%s: ', measures{i}); fprintf('\t%2.2f ', scores(i,:)); fprintf('\n');
%     end
end
end