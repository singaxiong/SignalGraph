
function [noisy, enhanced, clean, enhanced_wav, noisySTFT, mask, variance] = RunEnhanceNN(Data, layer, para)

output = FeatureTree2(Data, para, layer);

noisySTFT = gather(output{1}{1});
noisy = gather(output{1}{2});
enhanced = gather(output{1}{3});

enhanced_wav = abs2wav(exp(enhanced(1:257,:)/2)', angle(noisySTFT)', 400, 240);

if isempty(para.test.clean_idx)
    clean  = [];
else
    clean = gather(output{1}{para.test.clean_idx});
end

if isempty(para.test.mask_idx)
    mask = [];
else
    mask = output{1}{para.test.mask_idx};
end

if isempty(para.test.var_idx)
    variance = [];
else
    variance = output{1}{para.test.var_idx};
    if 0
        enhanced = enhanced - variance.^0.5;
        enhanced_wav = abs2wav(exp(enhanced(1:257,:)/2)', angle(noisySTFT)', 400, 240);
    end
    % [enhancedML,LL] = delta2static_ML(enhanced', variance', 2, 257);
end

        
end
