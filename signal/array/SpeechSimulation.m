% Given a sequence of clean sentences, a sequence of RIRs, and a sequence
% of noises, create a sequence of distorted speech
% Given a sequence of clean sentences, a sequence of RIRs that contains
% multiple sources, and a sequence of noises, create a sequence of
% distorted and mixed speech.

classdef SpeechSimulation
    properties
        nChannel = 1;   % number of microphone channels
        nSource = 1;    % number of speaker sources
        nSent2Simulate = 1;     % number of sentences to simulate
        precision = 'single';   % simulation for data storage
        useGPU = 0;
        
        maskType = 'none';      % none: do not generate time-frequency mask for output waveform. hard: generate 0/1 mask. soft: generate soft mask.
        randomizeFileOrder = 1; % whether to use random file order for clean stream
        gainNorm = 1;           % whether to normalize gain of output waveform
        
        fs = 16000;         % sampling rate
        frame_len = 400;    % window length when we need to perform FFT
        frame_shift = 160;  % window shift
        
        % parameters for segmenting output waveform into segments
        SegParm = struct('doSegmentation', 0, 'seglen', 100, 'segshift', 100);  % the unit of seglen and segshift is frame, usually there are 100 frames per second
        
        SNR_PDF =  struct('distribution', 'uniform', 'range', [0 20], 'mean', [], 'std', []);   % distribution of SNR.
        % if distribution is unifrom, provide the minimum and maximum SNRs.
        % if distribution is Gaussian, also provide mean and std.
        
        SPR_PDF = struct('distribution', 'uniform', 'range', [5 -5], 'mean', [], 'std', []);   % distribution of Signal power ratio (SPR), i.e. the ratio of power between sources
        
        cleanStream = [];   % a DataStream object that holds the clean sentences' file names of samples
        noiseStream = [];
        rirStream = [];
        vadStream = [];     % a DataStream object that holds the VAD file name or data for the cleanStream
        
        usedCleanStream = [];   % a DataStream object that holds the used clean sentences
        simulatedStream = [];   % a DataStream object that holds the simulated waveforms
        maskStream = [];        % a DataStream object that holds the time-frequency mask of the clean sentences in the simulated waveforms
        
        LOG = {};           % text that record the parameters used in simulation, e.g. SNR
    end
    
    methods
        function obj = SpeechSimulation(nChannel, nSource)
            obj.nChannel = nChannel;
            obj.nSource = nSource;
        end
        
        
        function [mixed_noisy, cleanWavAlignedScaled, mask, LOG] = SimulateMultiSourceOneSentence(obj, cleanIdx)   % simulate one sentence with specified clean source index
            
            % get the clean waveform
            if length(cleanIdx) ~= obj.nSource
                fprintf('Error: number of input clean sentence index not equal to planned number of sources\n'); return;
            end
            
            LOG = 'Clean:';
            
            for i=1:length(cleanIdx)
                tmp= obj.cleanStream.getData(cleanIdx(i), obj.precision);
                if ischar(obj.cleanStream.data{cleanIdx(i)})
                    LOG = [LOG ' ' obj.cleanStream.data{cleanIdx(i)}];
                else
                    LOG = [LOG ' NO_NAME'];
                end
                cleanWav{i} = tmp{1};
            end
            
            if isempty(obj.vadStream)
                vad = [];
            else
                for i=1:length(cleanIdx)
                    tmp = obj.vadStream.getData(cleanIdx(i), obj.precision); 
                    vad{i} = tmp{1};
                    nFrameVAD(i) = length(vad{i});
                end
            end
            
            if isempty(obj.rirStream)
                rirWav = [];
                LOG = sprintf('%s\tRIR: NULL', LOG);
            else
                % sample an RIR
                rirIdx = randperm(length(obj.rirStream.data));
                rirIdx = rirIdx(1);
                rirWav = obj.rirStream.getData(rirIdx, obj.precision); rirWav = rirWav{1};
                rirWav = rirWav /max(abs(rirWav(:)));       % normalize the gain of the RIR
                
                rirWav = reshape(rirWav, obj.nChannel, size(rirWav,1)/obj.nChannel, size(rirWav,2));
                
                if size(rirWav,2) < obj.nSource
                    fprintf('Error: number of sources in rir (%d) is smaller than the required sources (%d)\n', size(rirWav,2), obj.nSource);
                elseif size(rirWav,2) > obj.nSource
                    % sample required number of sources
                    randIdx = randperm(size(rirWav,2));
                    oldRirWav = rirWav;
                    rirWav = rirWav(:,randIdx(1:randIdx(obj.nSource)),:);
                end
                if ischar(obj.rirStream.data{rirIdx})
                    LOG = sprintf('%s\tRIR: %s', LOG, obj.rirStream.data{rirIdx});
                else
                    LOG = sprintf('%s\tRIR: NO_NAME', LOG);
                end

            end
            
            if isempty(obj.noiseStream)
                noiseWav = [];
                SNR = [];
                LOG = sprintf('%s\tNOISE: NULL', LOG);
            else
                % sample a noise file
                noiseIdx = randperm(length(obj.noiseStream.data));
                noiseIdx = noiseIdx(1);
                noiseWav = obj.noiseStream.getData(noiseIdx,obj.precision); noiseWav = noiseWav{1};
                
                % sample an SNR
                SNR = obj.SampleSNR(obj.SNR_PDF, 1);
                
                if ischar(obj.noiseStream.data{noiseIdx})
                    LOG = sprintf('%s\tNOISE: %s', LOG, obj.noiseStream.data{noiseIdx});
                else
                    LOG = sprintf('%s\tNOISE: NO_NAME', LOG);
                end
                LOG = sprintf('%s\tSNR: %2.4f', LOG, SNR);
            end
            
            % Simulate the source independently, then add the noises
            
            for i=1:obj.nSource
                [distorted{i}, reverb, direct{i}] = ApplyConstRirNoise(cleanWav{i}', obj.fs, squeeze(rirWav(:,i,:))', [], 0, obj.useGPU);
                distorted{i} = gather(distorted{i}(1:length(cleanWav{i}),:));
                direct{i} = gather(direct{i}(1:length(cleanWav{i}),:));
            end
            
            % Mix the sources together
            % Use the longest source as reference, sample the SPR of the
            % other sources
            SPR = obj.SampleSNR(obj.SPR_PDF, obj.nSource-1);
            LOG = sprintf('%s\tSPR:%s', LOG, sprintf(' %2.4f', [1 SPR(:)']));
            nSamples = cell2mat(cellfun(@size, distorted, 'UniformOutput', 0)');
            nSamples = nSamples(:,1);
            [~,order] = sort(nSamples, 'descend');
            mixed = distorted{order(1)};
            source1 = distorted{order(1)};
            
            mixer = SignalMixer();
            mixer.alignMethod = 'random_short_start';
            scale(order(1)) = 1;
            scaledSource{order(1)} = source1;
            startSampleIdx(order(1)) = 1;
            for i=2:obj.nSource
                [~, ~, scaledSource{order(i)}, scale(order(i)), tmpStartSampleIdx] = mixer.MixTwoSignals(source1, distorted{order(i)}, SPR(i-1));
                mixed = mixed + scaledSource{order(i)};
                startSampleIdx(order(i)) = tmpStartSampleIdx(2);
            end
            LOG = sprintf('%s\tScale:%s', LOG, sprintf(' %2.4f', scale));
            LOG = sprintf('%s\tStartSampleIdx:%s', LOG, sprintf(' %d', startSampleIdx));
            
            % add additive noise
            if isempty(noiseWav)
                mixed_noisy = mixed;
            else
                mixer.alignMethod = 'match_first_source';
                [mixed_noisy, ~, scaledNoise] = mixer.MixTwoSignals(mixed, noiseWav', SNR);
                if size(scaledNoise,2)==1   % if we are actually using single channel noise, add randomness to the noise in channels
                    randomness4noise = randn(size(scaledNoise,1), obj.nChannel);
                    randomness4noise = bsxfun(@times, randomness4noise, abs(scaledNoise)/10);    % scale the randomness by the noise itself.
                    % for noise sample with higher absolute amplitude, we use larger randomness and vice versa
                    scaledNoiseWithRandomness = bsxfun(@plus, randomness4noise, scaledNoise);
                    if 0
                        % verify that the noise in difference channels are
                        % different and look reasonable
                        [~,noise_spec] = wav2abs_multi(scaledNoiseWithRandomness, 16000);   % assume 16k sampling rate. It does not matter much for our purpose
                        noise_spec = noise_spec(1:257,:,:);
                        noise_spec_diff = bsxfun(@minus, noise_spec(:,1,:), noise_spec);
                        noise_spec_diff = reshape(noise_spec_diff, 257*obj.nChannel, size(noise_spec_diff,3));
                        % look at their amplitude, and phase differences
                        imagesc(real(noise_spec_diff));
                        imagesc(imag(noise_spec_diff));
                        imagesc(angle(noise_spec_diff));
                    end
                    mixed_noisy = mixed + scaledNoiseWithRandomness;    % replace mixed_noisy returned that is added with single channel noise
                end
            end
            
            frame_shift = length(cleanWav{1})/length(vad{1});
            if frame_shift>120; frame_shift = 160; else; frame_shift = 80; end

            for i = 1:obj.nSource
                cleanWavAlignedScaled{i} = zeros(max(nSamples),1);
                cleanWavAlignedScaled{i}(startSampleIdx(i):startSampleIdx(i)+length(distorted{i})-1) = cleanWav{i}' * scale(i);
                % generate new VAD
                vadAligned{i} = zeros(max(nFrameVAD),1);
                frame_offset = round(startSampleIdx(i)/frame_shift);
                vadAligned{i}(frame_offset+1:frame_offset+length(vad{i})) = vad{i};
            end
                
            if strcmpi(obj.maskType, 'none')
                mask = [];
            else
                softMask = strcmpi(obj.maskType, 'soft');
                % it's better to use direct, which contains the early reflection up to 50ms after the direct sound, as the clean reference.
                % roughly decide frame rate
                for i = 1:obj.nSource
                    curr_direct = zeros(max(nSamples),1);
                    curr_direct(startSampleIdx(i):startSampleIdx(i)+length(direct{i})-1) = direct{i} * scale(i);
                    mask{i} = genMaskFromParallelData(cleanWavAlignedScaled{i}, curr_direct, mixed, vadAligned{i}, obj.fs, softMask, 0,0);
                end
            end
            
            mixed_noisy = mixed_noisy';
            if obj.gainNorm
                max_mixed = max(abs(mixed_noisy(:)));
                mixed_noisy = mixed_noisy / max_mixed;
                for i=1:obj.nSource
                    cleanWavAlignedScaled{i} = cleanWavAlignedScaled{i}' / max_mixed;
                end
            end
            
            
            
        end
        
        function [distorted, cleanWav, mask] = SimulateSingleSourceOneSentence(obj, cleanIdx)   % simulate one sentence with specified clean source index
            
            % get the clean waveform
            cleanWav = obj.cleanStream.getData(cleanIdx, obj.precision); cleanWav = cleanWav{1};
            
            if isempty(obj.vadStream)
                vad = [];
            else
                vad = obj.vadStream.getData(cleanIdx, obj.precision); vad = vad{1};
            end
            
            if isempty(obj.rirStream)
                rirWav = [];
            else
                % sample an RIR
                rirIdx = randperm(length(obj.rirStream.data));
                rirIdx = rirIdx(1);
                rirWav = obj.rirStream.getData(rirIdx, obj.precision); rirWav = rirWav{1};
                rirWav = rirWav /max(abs(rirWav(:)));       % normalize the gain of the RIR
            end
            
            if isempty(obj.noiseStream)
                noiseWav = [];
                SNR = [];
            else
                % sample a noise file
                noiseIdx = randperm(length(obj.noiseStream.data));
                noiseIdx = noiseIdx(1);
                noiseWav = obj.noiseStream.getData(noiseIdx,obj.precision); noiseWav = noiseWav{1};
                
                % sample an SNR
                SNR = obj.SampleSNR(obj.SNR_PDF, 1);
            end
            
            [distorted, reverb, direct] = ApplyConstRirNoise(cleanWav', obj.fs, rirWav', noiseWav', SNR, obj.useGPU);
            distorted = gather(distorted(1:length(cleanWav),:));
            direct = gather(direct(1:length(cleanWav),:));
            
            if strcmpi(obj.maskType, 'none')
                mask = [];
            else
                softMask = strcmpi(obj.maskType, 'soft');
                % it's better to use direct, which contains the early reflection up to 50ms after the direct sound, as the clean reference.
                mask = genMaskFromParallelData(cleanWav', direct, distorted, vad, obj.fs, softMask, 0, 0);
            end
            
            distorted = distorted';
            if obj.gainNorm
                distorted = distorted / max(abs(distorted(:)));
            end
        end
        
        function SimulateSingleSource(obj, nSent2Simulate)
            cleanIdx;
            SimulateSingleSourceOneSentence(cleanIdx);
            
        end
        
    end
    
    methods (Access = protected)
        function SNR = SampleSNR(obj, SNR_PDF, nSNR)
            switch lower(SNR_PDF.distribution)
                case 'uniform'
                    SNR = rand(nSNR,1) * (SNR_PDF.range(2) - SNR_PDF.range(1)) + SNR_PDF.range(1);
                case 'normal'
                    for i=1:nSNR
                        for ii=1:100
                            currSNR = randn(1) * SNR_PDF.std + SNR_PDF.mean;
                            if currSNR>=SNR_PDF.range(1) && currSNR<=SNR_PDF.range(2)
                                break;
                            end
                        end
                        if currSNR<SNR_PDF.range(1) || currSNR>SNR_PDF.range(2)     % if we failed get acceptable SNR, just sample with uniform distribution
                            currSNR = rand(1) * (SNR_PDF.range(2) - SNR_PDF.range(1)) + SNR_PDF.range(1);
                        end
                        SNR(i) = currSNR;
                    end
                otherwise
                    fprintf('Unknown SNR distribution: %s\n', SNR_PDF.distribution);
                    return;
            end
        end
    end
end
