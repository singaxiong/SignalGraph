% This class defines how a graph is updated.
% It manages learning rate decay, momentum, gradient and weigth clipping,
% etc.

classdef GraphLearnerSGD
    properties
        % learning rate scheme
        learningRateDecayScheme = 'expDecay';    % [expDecay | decayIfNoImprovement]
        initLearningRate = 1e-2;
        minLearnRate = 1e-4;
        learningRate = 1e-2;
        learningRateDecayPerMilestone = 0.999;
        learningRateDecayPerItr = 0.999;
        nHourSeen = 0;
        reduceLREveryXHour = 0.5;
        nextMilestone = 0.5;
        
        % momentum scheme
        momentumIncreaseScheme= 'exp';
        initMomentum = 0;
        maxMomentum = 0.9;
        momentum = 0;
        momentumIncreaseSpeed = 1.01;
        
        rmspropDecay = 0;   % reserved for future
        
        gradientClipThreshold = 0;
        weightClipThreshold = 0;
        
        monitorWeightNorm = 0;
        
        % cache of gradients in previous minibatch
        updateHistory;
    end
    methods
        function obj = GraphLearnerSGD(sigGraph)
            % initialize the update structure used for momentum
            for k=1:length(sigGraph.WeightUpdateOrder)
                obj.updateHistory{k}.W = 0;
                obj.updateHistory{k}.b = 0;
            end
        end
        
        function obj = accumulateTrainDataDuration(obj, nHourSeen)
            obj.nHourSeen = obj.nHourSeen + nHourSeen;
        end
        
        function obj = updateLearningRateEndOfBlock(obj)
            if obj.nHourSeen > obj.nextMilestone
                obj.learningRate     = obj.learningRate * obj.learningRateDecayPerMilestone;   % Learning rate for biases of hidden units
                fprintf('Trained with %2.2f hours of data, reducing learning rate to %f\n', obj.nHourSeen, obj.learningRate);
                obj.nextMilestone = obj.nextMilestone + obj.reduceLREveryXHour;
            end
        end
        
        function obj = updateLearningRateEndOfIteration(obj)
            switch obj.learningRateDecayScheme
                case 'decayIfNoImprovement'     % if CV cost does not decrease, reduce the learning rate
                    % to be implemented
                case 'expDecay'     % decay the learning rate no matter what happens to the CV cost. Run until a preset number of iterations is finished.
                    obj.learningRate = obj.learningRate * obj.learningRateDecayPerItr;
            end
        end
        function sigGraph = updateGraph(obj, sigGraph)
            WeightUpdateOrder = sigGraph.WeightUpdateOrder;
            layer = sigGraph.layer;
            curr_momentum = obj.momentum;
            
            total_weight_norm = 0;
            
            if isnan(layer{end}.a)
                % Sometimes, we get nan cost function. Then we should ignore current mismatch.
                % Cases that causes nan cost function: 1) in single precision, if the
                % input of softmax is too big, it will cause nan as the exp(input) is inf.
                fprintf('Warning: nan cost detected, abort updating graph!\n');
                return;
            end
            
            for i=1:length(WeightUpdateOrder)
                Lidx = WeightUpdateOrder{i};
                [~, isTranspose] = obj.VerifyTiedLayers(layer(Lidx));
                
                % collect gradients
                gradW = layer{Lidx(1)}.gradW;
                if sum(isnan(gradW(:)))
                    pause(0.1);
                end
                for k=2:length(Lidx)
                    if isTranspose(k)
                        gradW = gradW + layer{Lidx(k)}.gradW';
                    else
                        gradW = gradW + layer{Lidx(k)}.gradW;
                    end
                end
                
                if obj.gradientClipThreshold > 0
                    gradW = max(-obj.gradientClipThreshold, gradW);
                    gradW = min(obj.gradientClipThreshold, gradW);
                end
                
                %     if obj.rmspropDecay > 0
                %         layer{k}.gradW_avg_square = layer{k}.gradW_avg_square * para.rmsprop_decay + ...
                %             layer{k}.gradW.^2 * (1-para.rmsprop_decay);
                %         element_learning_rate = 1./(sqrt(layer{k}.gradW_avg_square)+para.rmsprop_damping);
                %         element_learning_rate = element_learning_rate / ...
                %             sum(sum(element_learning_rate))*numel(element_learning_rate);
                %         update{k}.W = update{k}.W * curr_momentum + ...
                %             gradW.*element_learning_rate * learning_rate;
                %     else
                if issparse(gradW)==0      % apply momentum only when gradient is not sparse
                    obj.updateHistory{i}.W =obj.updateHistory{i}.W * curr_momentum + gradW * obj.learningRate;
                else
                    obj.updateHistory{i}.W = gradW * learning_rate;
                end
                
                if obj.monitorWeightNorm
                    weight_norm_old = mean(mean(layer{Lidx(1)}.W.^2)) * length(Lidx);
                end
                
                if issparse(obj.updateHistory{i}.W)
                    layer{Lidx(1)}.W = AddSpMatMat(-1,obj.updateHistory{i}.W, 1, layer{Lidx(1)}.W, 0);
                else
                    layer{Lidx(1)}.W = layer{Lidx(1)}.W - obj.updateHistory{i}.W;
                end
                
                if obj.weightClipThreshold > 0
                    % sometimes the weight will explode, so we need to add a limit to the value of the weights, e.g. +-10
                    layer{Lidx(1)}.W = max(-obj.weightClipThreshold,layer{Lidx(1)}.W);
                    layer{Lidx(1)}.W = min(obj.weightClipThreshold,layer{Lidx(1)}.W);
                end
                
                for k=2:length(Lidx)   % copy weights to other tied layers
                    if isTranspose(k)
                        layer{Lidx(k)}.W = layer{Lidx(1)}.W';
                    else
                        layer{Lidx(k)}.W = layer{Lidx(1)}.W;
                    end
                end
                
                if obj.monitorWeightNorm
                    weight_norm = mean(mean(layer{Lidx(1)}.W.^2)) * length(Lidx);
                    total_weight_norm = total_weight_norm + weight_norm;
                    if weight_norm/weight_norm_old > 1.5
                        fprintf('Warning: layer %d weight norm increases too fast: old norm: %f, new norm %f\n',k,weight_norm_old, weight_norm);
                    end
                end
                
                has_bias = isprop(layer{Lidx(1)}, 'gradB');
                
                if has_bias
                    gradB = layer{Lidx(1)}.gradB;
                    for k=2:length(Lidx)
                        if ~isTranspose(k)      % if the layer is a transpose of first layer, its gradB is not used and its b won't be trained
                            gradB = gradB + layer{Lidx(k)}.gradB;
                        end
                    end
                    if obj.rmspropDecay > 0
                        %             layer{k}.gradb_avg_square = layer{k}.gradb_avg_square * para.rmsprop_decay + ...
                        %                 layer{k}.gradB.^2 * (1-para.rmsprop_decay);
                        %             element_learning_rate = 1./(sqrt(layer{k}.gradb_avg_square)+para.rmsprop_damping);
                        %             element_learning_rate = element_learning_rate / ...
                        %                 sum(sum(element_learning_rate))*numel(element_learning_rate);
                        %            obj.updateHistory{k}.b =obj.updateHistory{k}.b * curr_momentum + ...
                        %                 layer{k}.gradB.*element_learning_rate * learning_rate;
                    else
                        if curr_momentum>0
                            obj.updateHistory{i}.b =obj.updateHistory{i}.b * curr_momentum + gradB * obj.learningRate;
                        else
                            obj.updateHistory{i}.b = gradB * obj.learningRate;
                        end
                    end
                    layer{Lidx(1)}.b = layer{Lidx(1)}.b -obj.updateHistory{i}.b;
                    for k=2:length(Lidx)   % copy biases to other tied layers
                        if ~isTranspose(k)
                            layer{Lidx(k)}.b = layer{Lidx(1)}.b;
                        end
                    end
                end
            end
            sigGraph.layer = layer;
        end
        
    end
    methods (Access = protected)
        function [dimMismatch, isTranspose] = VerifyTiedLayers(obj,tiedLayers)
            
            for j=1:length(tiedLayers)
                dim(:,j) = tiedLayers{j}.dim;
            end
            dimMismatch = [];
            isTranspose = [];
            for j=2:length(tiedLayers)
                if sum(abs(dim(:,j)-dim(:,1)))
                    if sum(abs(dim(end:-1:1,j)-dim(:,1)))
                        dimMismatch(j) = 1;
                        fprintf('Error: dimension mismatch between layers that share the same weight matrix\n');
                    else
                        isTranspose(j) = 1;
                    end
                else
                    isTranspose(j) = 0;
                end
            end
            
        end
    end
end
