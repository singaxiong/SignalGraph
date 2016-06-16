function [sd, tn] = delaySignal(s,fs,d,ns,method)

% Purpose : Generate a delayed version of the signal in sourceSignal.


%  Find length and deminsion of input signal
[r,c] = size(s);

%  Covert to column vecotors for processing
if r == 1;
    s = s.';  %  Ensure it is a column vector
    [rn,c] = size(s);
elseif c == 1
    [rn,c] = size(s); % Otherwise rn = r indicating it came in as a column vector
else   %  If multi column and row then process each column as its own signal
    [rn,c] = size(s); % Otherwise input was columnwise matrix
    if length(d) ~= c
        error('Number of columns in signal matrix must equal number of elements in delay vector')
    end;
end;

if min(d) < 0;
    error('Delay cannot be negative')
end
% Compute requested delay in sample points rounded off to nearest sample
nd = ceil(d*fs);

%  Determine final output length
if nargin == 4
    slen = ceil(ns*fs);
else
    slen = max(nd)+rn;   %  Take max of all delays in vector to determine final signal length
end
sd = zeros(slen,c);       % Initialize output matrix

switch upper(method)

    case 'SIMPLEDELAY'
        
        sdd = zeros(slen,1);  % Initalize dummy vector for storing integer shift

        %  Loop through each row of signal matrix and apply delay
        for k=1:c
            %  Shift integer sample component of delay
            id = fix(d(k)*fs)+1;
            sdd(id:min([slen,(rn+id-1)])) = s(1:min([slen-id+1, rn]),k);
            sd(:,c-k+1) = sdd(1:slen);
        end

        %  Determine final output length

    case 'FFTDELAY'
        if slen <= rn;
            sd = s(1:slen,:);
        else
            sd(1:rn,:) = s;
        end;


        %  Create frequency shift vector in the frequency domain
        nfft = 2^nextpow2(2*slen);
        fax = fs*(-nfft/2:nfft/2-1)'/nfft;

        %  Loop through each column of signal matrix and apply delay
        for k=1:c
            shft = exp(-j*d(k)*2*pi*fax);   % Frequency function for delay
            shft = ifftshift(shft);         % Make axis compatable with numeric FFT
            fsd = fft(sd(:,k),nfft);        % Take FFT
            fsd = fsd.*shft;                %  Apply delay
            dum = ifft(fsd);                %  Return to time domain
            sd(:,k) = dum(1:slen);          %  Trim time domain signal to required length
        end

    otherwise error('delay method not defined...');

end;

%  Restore dimention of signal vector to original orientation
if rn == r     % If input was originally a column or multi-signal matrix we are done
    if nargout == 2      %   Create time axis if requested
        tn = [0:slen-1]'/fs;
    end
else      % If input was originally a row vector, take transpose
    sd = (sd.');
    if nargout == 2      %   Create time axis if requested
        tn = [0:slen-1]/fs;
    end
end

% if isreal(s)
%     sd = real(sd);
% end
