function output = F_Itakura_Saito(input_layers)
[m, output, target] = prepareCostEvaluation(input_layers);
        % here we assume that the output of the DNN and the reference are
        % both in log spectrum domain, but we would like to use the IS
        % distortion measure in linear spectrum domain. The distance is
        % defined as d_is(X|Y) = exp(X-Y) - (X-Y) + 1, where X is the
        % target and Y is the DNN output, both in log domain. 
        % Note that IS distortini s not symmetrical.
        diff = target-output;
        cost = exp(diff) - diff - 1;
        cost = 1/m * sum( sum( cost) );
end