% Implement the forward pass of an LSTM layer. 
% Author: Xiong Xiao, Temasek Labs, NTU, Singapore. 
% Last modified: 13 Oct 2015
%
function [LSTM_layer] = F_LSTM(input_layer, LSTM_layer)
%
% The weight matrix of LSTM is organized as follows:
% W = [ W_cf W_hf W_xf;
%       W_cc W_hc W_xc;
%       W_ci W_hi W_xi;
%       W_co W_ho W_xo;]
% where W_cf is the weights that connects past cell states to
% forget gates, W_hc is the weights between past hidden activation
% to candidate states, W_xi is the weights beween input features to
% input gates. Other weights are similarly defined.
%
% We introduce a vector at = [ft; Ct_raw; it; ot], and zt is the vector of at
% before the sigmoid or tanh activation functions. We also introduce matrix 
% W = [Wc Wh Wx]; where Wc = [W_cf; W_cc; W_ci; W_co] and so on. 
% Let st = [Ct-1 ht-1 input]. Then, the LSTM will first compute
% zt = W * st. 

% Wc and Wh are 4N x N matrices, where N is the number of cells in the
% layer. Wx is a 4N x dim matrix, where dim is the number of inputs.

% input are multiple trajectories, usually of different lengths. We need to
% first find the number of effective frames

input = input_layer.a;
[dim, nFr, nSeg] = size(input);
if nSeg>1
    [mask, variableLength] = getValidFrameMask(input_layer);
    if variableLength; input = PadShortTrajectory(input, mask, 0); end
else
    mask = [];
end
LSTM_layer.validFrameMask = mask;

input = permute(input, [1 3 2]);
if strcmpi(class(input), 'gpuArray'); useGPU=1; else useGPU = 0; end
precision = class(gather(input(1)));

useHidden = 1;
if isfield(LSTM_layer, 'usePastState')
    usePastStateAsFeature = LSTM_layer.usePastState;
else
    usePastStateAsFeature=0;
end
usePastState = 1;

nCell = LSTM_layer.dim(1);  % number of LSTM cells in the layer

if usePastStateAsFeature
    Wc = LSTM_layer.W(:,1:nCell);           % weights of past cell states. Note that the second block of Wc corresponding to Ct_raw is zero.
    Wh = LSTM_layer.W(:,nCell+1:nCell*2);   % weights of past hidden activations
    Wx = LSTM_layer.W(:,nCell*2+1:end);     % weights of current input vector
else
    Wh = LSTM_layer.W(:,1:nCell);   % weights of past hidden activations
    Wx = LSTM_layer.W(:,nCell+1:end);     % weights of current input vector
end
b = LSTM_layer.b;     % bias

% initialize LSTM state vector and hidden layer output
if useGPU==0
    Ct0 = ones(nCell,nSeg, precision);   % initial cell states
    ht0 = ones(nCell,nSeg, precision);   % initial hidden layer output
    % allocate memory for gates and states
    ft = zeros(nCell, nSeg, nFr, precision);     % forget gates
    it = zeros(nCell, nSeg, nFr, precision);     % input gates
    ot = zeros(nCell, nSeg, nFr, precision);     % output gates
    Ct_raw = zeros(nCell, nSeg, nFr, precision); % candidate cell states
    Ct = zeros(nCell, nSeg, nFr, precision);     % cell states
    ht = zeros(nCell, nSeg, nFr, precision);     % hidde layer output, i.e. the output of the LSTM layer
else
    Ct0 = gpuArray.ones(nCell,nSeg, precision);   % initial cell states
    ht0 = gpuArray.ones(nCell,nSeg, precision);   % initial hidden layer output
    % allocate memory for gates and states
    ft = gpuArray.zeros(nCell, nSeg, nFr, precision);     % forget gates
    it = gpuArray.zeros(nCell, nSeg, nFr, precision);     % input gates
    ot = gpuArray.zeros(nCell, nSeg, nFr, precision);     % output gates
    Ct_raw = gpuArray.zeros(nCell, nSeg, nFr, precision); % candidate cell states
    Ct = gpuArray.zeros(nCell, nSeg, nFr, precision);     % cell states
    ht = gpuArray.zeros(nCell, nSeg, nFr, precision);     % hidde layer output, i.e. the output of the LSTM layer
end
    
% batch transform the input features for fast speed
fakeLayer.a = input;
fakeLayer.validFrameMask = mask;
z_from_inputs = F_affine_transform(fakeLayer, Wx, b);

for i=1:nFr
    if i==1     % for the first frame, use default values for past state and hidden values.
        Ct_past = Ct0;
        ht_past = ht0;
    else
        Ct_past = Ct(:,:, i-1);
        ht_past = ht(:,:, i-1);
    end

    zt = z_from_inputs(:,:, i);
    if usePastStateAsFeature
        zt = zt + Wc * Ct_past;
    end
    if useHidden
        zt = zt + Wh * ht_past;
    end

    % extract the elements of zt to compute the gates
    zt_sig = sigmoid(zt);
    ft(:,:,i) = zt_sig(1:nCell,:);
    Ct_raw(:,:,i)    = tanh(     zt(nCell+1:nCell*2,:)   );
    it(:,:,i) = zt_sig(nCell*2+1:nCell*3,:);
    ot(:,:,i) = zt_sig(nCell*3+1:nCell*4,:);
    
    % compute the states of the cells, which is a weighted sum of the
    % current state and past state. 
    if usePastState
        Ct(:,:,i) = Ct_raw(:,:,i) .* it(:,:,i) + Ct_past .* ft(:,:,i);
    else
        Ct(:,:,i) = Ct_raw(:,:,i) .* it(:,:,i);
    end
    
    % compute the output
    ht(:,:,i) = tanh(Ct(:,:,i)) .* ot(:,:,i);
end

if nSeg>1 && variableLength; 
    ht = permute(PadShortTrajectory(permute(ht,[1 3 2]), mask, -1e10), [1 3 2]);
    ft = permute(PadShortTrajectory(permute(ft,[1 3 2]), mask, 0), [1 3 2]);
    it = permute(PadShortTrajectory(permute(it,[1 3 2]), mask, 0), [1 3 2]);
    ot = permute(PadShortTrajectory(permute(ot,[1 3 2]), mask, 0), [1 3 2]);
    Ct_raw = permute(PadShortTrajectory(permute(Ct_raw,[1 3 2]), mask, 0), [1 3 2]);
    Ct = permute(PadShortTrajectory(permute(Ct,[1 3 2]), mask, 0), [1 3 2]);
end

LSTM_layer.ft = ft;
LSTM_layer.it = it;
LSTM_layer.ot = ot;
LSTM_layer.Ct_raw = Ct_raw;
LSTM_layer.Ct = Ct;
LSTM_layer.a = permute(ht, [1 3 2]);  
LSTM_layer.Ct0 = Ct0;
LSTM_layer.ht0 = ht0;

end
