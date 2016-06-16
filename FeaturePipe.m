function output = FeaturePipe(input, processing, input2)

output = input;
for i=1:length(processing)
    switch processing{i}.name
        
        % data manipulations

        case 'CPU2GPU'
            output = gpuArray(output);
        case 'GPU2CPU'
            output = gather(output);
        case 'selectDim'
            output = output(processing{i}.transform,:);
        case {'splice', 'Splice'}
            output = ExpandContext_v2(output, processing{i}.transform);
        case 'stream2'  % extra input
            output = [output; input2];
        case 'segmentation'     % this is similar to enframe. It is usually used for high dimensional input, while enframe is usually for single/multichannel waveforms
            seglen = processing{i}.transform(1);
            segshift = processing{i}.transform(2);
            % if input is 1D array, output is 2D matrix. If input is 2D matrix, output is 3D tensor.The last dimension is the number of segments
            % Any remaining frames that is shorter than seglen is discarded.
            output = DivideSent2Segments(output, seglen, segshift, 0);  
        case 'transpose'
            output = output';
        case 'permute'
            output = permute(output, processing{i}.transform);
        case 'vectorize'
            output = reshape(output,numel(output),1);
            
        case 'removeNonspeechFrame' % assume the input is something related to energy. Do energy based VAD first, then discard those frames classified as nonspeech
            
        case 'upsample'
            
        case 'downsample'

            % signal processing
            
        case 'addNoise'
            output = output + randn(size(output)) * processing{i}.transform;
        case 'removeDC'
            output = DC_remove(output,0.999);
        case 'preemphasis'
            output = filter([1 -0.97],1,output);
        case 'enframe'
            output = my_enframe(output, processing{i}.transform(1), processing{i}.transform(2));
        case 'windowing'
            switch processing{i}.transform
                case 'hamming'
                    window = hamming(frame_size);
                case 'hanning'
                    window = hanning(frame_size);
                otherwise
                    window = [];
            end
            if ~isempty(window)
                output = bsxfun(@times, output, window);
            end
        case 'fft'
            output = fft(output,processing{i}.transform);
        case 'log'
            if isfield(processing{i}, 'transform')
                output = log(output+processing{i}.transform);
            else
                output = log(output);
            end
        case 'power'
            output = real(output .* conj(output));
        case 'complex2realImag'
            [D,nFr] = size(output);
            output = output(1:D/2,:) + sqrt(-1)*output(D/2+1:end,:);
        case 'realImag2complex'
            output = [real(output); imag(output)];
            
            % language processing
            
        case 'seq2ngram'
            output = seq2ngram(output, processing{i}.ngram, processing{i}.vocab);
        case 'idx2vec'
            output2 = sparse(processing{i}.outputDim, length(output));
            for j=1:length(output)
                output2(output(j),j)= 1;
            end
            output = output2;
        case {'context_sum', 'Context_sum'}
            context = processing{i}.transform;
            output = SumContext(output, context);
        case 'fulltify'
            output = full(output);
            
            % temporal processing
            
        case 'delta'
            output = comp_dynamic_feature(output', processing{i}.delta_order, processing{i}.delta_order)';
        case 'dynamic'
            D = genDeltaTransform(size(output,2), 2);
            A = D*D;
            output = [output; output*D'; output*A'];
            
            % neural networks steps
            
        case {'affinetransform','AffineTransform'};
            output = processing{i}.transform * output;
            if isfield(processing{i}, 'bias')
                bias = processing{i}.bias;
                output = bsxfun(@plus, output, bias(:));
            end
        case {'sigmoid','Sigmoid'}
            output = sigmoid(output);
        case {'softmax', 'Softmax'}
            output = softmax(output);
        case {'addshift', 'AddShift'}
            output = bsxfun(@minus, output, processing{i}.transform');
        case {'rescale', 'Rescale'}
            if issparse(output)
                visible_nonzero_idx = find(sum(abs(output),2)>0);
                visible_nonzero = full(output(visible_nonzero_idx,:));
                output(visible_nonzero_idx) = bsxfun(@times, visible_nonzero, processing{i}.transform(visible_nonzero_idx)');
            else
                output = bsxfun(@times, output, processing{i}.transform');
            end
        case 'linear'
            % linear activation node, do nothings
        case 'phoneID2posterior'
            output = phoneID2posterior(output, length(processing{i}.classID), processing{i}.classID);
        case 'phoneID2classID'
            output = phoneID2classID(output, processing{i}.classID);
 
            % normalization methods
            
        case 'MVN'
            output = MVN(output')';
        case 'CMN'
            output = CMN(output')';
        case 'length_norm'
            scale = 1./sqrt(sum(output.^2,1));
            output = bsxfun(@times, output, scale);

        otherwise
            fprintf('Error: unknown processing step: %s\n', processing{i}.name);
            break;
    end
end
end