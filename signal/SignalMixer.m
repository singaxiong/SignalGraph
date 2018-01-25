classdef SignalMixer
    
    properties
        alignMethod = 'random_short_start';   % how to deal with different lengths of the sources.
        % random_short_strat: randomly select a starting point for the shorter source
        % repeat_short: repeat the shorter source several times
        % random_cut_long: randomly select a portion of the long source
        % match_first_source: always match the length of the first source
        overlapPercentage = 1;  % to be implemented: percentage of overlap. Better to have VAD information
    end
    
    methods
        function obj = SignalMixer()
            
        end
        function [mixed, component1, component2, scale2, startSampleIdx] = MixTwoSignals(obj, wav1, wav2, SPR)
            
            [n1,nCh1] = size(wav1);
            [n2,nCh2] = size(wav2);
            if nCh1 ~= nCh2 && nCh1>1 && nCh2 > 1
                fprintf('Error: number of channels are different in sources\n'); 
                return;
            end
            
            % scale wav2 to obtain desired SPR, the signal power ratio in dB
            power1 = mean(wav1(:).^2);
            power2 = mean(wav2(:).^2);
            scale2 = sqrt(power1/power2 * 10^(-SPR/10));
            wav2 = scale2*wav2;      % becarefull that the mixed waveform may need to be scaled to avoid clipping
            
            switch lower(obj.alignMethod)
                case 'repeat_short'
                    % if the two waveforms have different length, repeat the shorter one to
                    % match the longer one
                    if n1>n2
                        component1 = wav1;
                        component2 = obj.MatchReferenceLengthRepeat(wav1, wav2);
                    else
                        component1 = obj.MatchReferenceLengthRepeat(wav2, wav1);
                        component2 = wav2;
                    end
                case 'random_short_start'
                    [component1,component2, startSampleIdx] = obj.MatchReferenceLengthAppendZeros(wav1, wav2);
                case 'random_cut_long'
                    if n1<n2
                        component1 = wav1;
                        component2 = obj.MatchReferenceLengthCut(wav1, wav2);
                    else
                        component2 = wav2;
                        component1 = obj.MatchReferenceLengthCut(wav2, wav1);
                    end
                case 'match_first_source'
                    component1 = wav1;
                    component2 = obj.MatchReferenceLengthRepeat(wav1, wav2);
            end
            
            mixed = bsxfun(@plus, component1, component2);
            
        end
    end
    methods (Access = protected)
        % if wav is longer than ref, take a random segment of wav that
        % matches the length of ref. 
        % if wav is shorter than ref, first repeat wav several times. 
        function matched_wav = MatchReferenceLengthRepeat(obj, ref, wav)
            [n1,nCh] = size(ref);
            [n2,nCh] = size(wav);
            if n1==n2
                matched_wav = wav;
                return;
            end
            
            nRepeat = ceil(n1/n2);
            repeated_wav = repmat(wav, nRepeat,1);
            
            matched_wav = obj.MatchReferenceLengthCut(ref, repeated_wav);
        end
        
        % assume ref is shorter than wav, randomly cut a segment of wav to
        % match the length of ref
        function matched_wav = MatchReferenceLengthCut(obj, ref, wav)
            [n1,nCh] = size(ref);
            [n2,nCh] = size(wav);
            if n1==n2
                matched_wav = wav;
                return;
            end
            
            gap = n2 - n1;
            offset = floor(rand(1)*gap);
            matched_wav = wav(offset+1:offset+n1,:);
        end
        
        % assume ref is longer than wav, we want to append random number of 
        % zeros to the both ends of wav to match the length of ref
        % if ref is shorter than wav, switch their position. 
        function [ref, matched_wav, startSampleIdx] = MatchReferenceLengthAppendZeros(obj, ref, wav)
            [n1,nCh] = size(ref);
            [n2,nCh] = size(wav);
            if n1==n2
                matched_wav = wav;
                startSampleIdx = [1 1];
                return;
            elseif n1<n2
                [matched_wav,  ref] = obj.MatchReferenceLengthAppendZeros(wav,ref);
                startSampleIdx = startSampleIdx([2 1]);
                return;
            end
            
            gap = n1-n2;
            offset = floor(rand(1)*gap);
            matched_wav = zeros(n1,nCh);          % there will be offset zeros before wav, and gap-offset zeros after wav
            matched_wav(offset+1:offset+n2,:) = wav;
            startSampleIdx(1) = 1;
            startSampleIdx(2) = offset+1;
        end
    end
    
end
