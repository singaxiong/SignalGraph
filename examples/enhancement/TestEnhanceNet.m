function TestEnhanceNet()
addpath('local');

dnn_files{1} = 'nnet/EnhanceRegression.noCMN.DeltaByEqn.MbSize40.U28539.771-LSTM-2048-771.L2_3E-4.LR_3E-3/nnet.itr37.LR1E-5.CV2453.828.mat';
dnn_files{2} = 'nnet/EnhanceRegression.noCMN.DeltaByEqn.MbSize20.U28539.771-LSTM-2048-771.L2_3E-4.LR_1E-4/nnet.itr62.LR4.97E-8.CV2439.245.mat';
dnn_files{3} = 'nnet/EnhanceGaussian.noCMN.init1.expDecay.Decay0.999.DeltaByEqn.MbSize40.U28539.771-LSTM-2048-771.L2_0.LR_3E-3/nnet.itr26.LR3.62E-5.CV756.273.mat';
IsMaskNet = [0 0 0];

testset = 2;
T60 = [0.01 0.3 0.6];
noise = {'buccaneer2', 'pink', 'm109', 'hfchannel', 'factory2', 'factory1', 'destroyerops', 'babble', 'volvo', 'leopard', 'buccaneer1', 'destroyerengine', 'f16', 'machinegun', 'white'};
% noise = {'f16', 'machinegun', 'white'};
SNR = [10 0 -10];
DEBUG = 0;
measures = {'PESQ', 'FWSEQ', 'CD', 'IS', 'LLR', 'WSS'};
useGPU = 0;

for t60_i = 2%:length(T60)
    for n_i = 12%:length(noise)
        for s_i = 3%:length(SNR)
            TestEnhanceNetByCategory(dnn_files, testset, T60(t60_i), noise{n_i}, SNR(s_i), IsMaskNet, measures, useGPU, DEBUG)
        end
    end
end

end
