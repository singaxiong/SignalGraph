function [f,t,w]=enframe(x,win,inc,m)
%ENFRAME split signal up into (overlapping) frames: one per row. [F,T]=(X,WIN,INC)
%
% Usage:  (1) f=enframe(x,n)     % split into frames of length n
%         (2) f=enframe(x,hamming(n,'periodic'),n/4)     % use a 75% overlapped Hamming window of length n
%         (3) calculate spectrogram in units of power per Hz
%
%               W=hamming(NW);                      % analysis window (NW = fft length)
%               W=W/sqrt(FS*sum(W.^2));             % normalize to give power per Hz (FS = sample freq)
%               P=rfft(enframe(S,W,INC);,nfft,2);   % computer first half of fft (INC = frame increment in samples)
%               P(:,2:end-1)=2*P(:,2:end-1);        % double to account for -ve frequencies (except DC and Nyquist)
%
%         (3) frequency domain frame-based processing:
%
%               S=...;                              % input signal
%               OV=2;                               % overlap factor of 2 (4 is also often used)
%               INC=20;                             % set frame increment in samples
%               NW=INC*OV;                          % DFT window length
%               W=sqrt(hamming(NW,'periodic'));     % omit sqrt if OV=4
%               W=W/sqrt(sum(W(1:INC:NW).^2));      % normalize window
%               F=rfft(enframe(S,W,INC),NW,2);      % do STFT: one row per time frame, +ve frequencies only
%               ... process frames ...
%               X=overlapadd(irfft(F,NW,2),W,INC);  % reconstitute the time waveform (omit "X=" to plot waveform)
%
%  Inputs:   x    input signal
%          win    window or window length in samples
%          inc    frame increment in samples
%            m    mode input:
%                  'z'  zero pad to fill up final frame
%                  'r'  reflect last few samples for final frame
%                  'A'  calculate the t output as the centre of mass
%                  'E'  calculate the t output as the centre of energy
%
% Outputs:   f    enframed data - one frame per row
%            t    fractional time in samples at the centre of each frame
%                 with the first sample being 1.
%            w    window function used
%
% By default, the number of frames will be rounded down to the nearest
% integer and the last few samples of x() will be ignored unless its length
% is lw more than a multiple of inc. If the 'z' or 'r' options are given,
% the number of frame will instead be rounded up and no samples will be ignored.
%
% Example of frame-based processing:
%          INC=20       						% set frame increment in samples
%          NW=INC*2     						% oversample by a factor of 2 (4 is also often used)
%          S=cos((0:NW*7)*6*pi/NW);				% example input signal
%          W=sqrt(hamming(NW),'periodic'));  	% sqrt hamming window of period NW
%          F=enframe(S,W,INC);               	% split into frames
%          ... process frames ...
%          X=overlapadd(F,W,INC);               % reconstitute the time waveform (omit "X=" to plot waveform)

% Bugs/Suggestions:
%  (1) Possible additional mode options:
%        'u'  modify window for first and last few frames to ensure WOLA
%        'a'  normalize window to give a mean of unity after overlaps
%        'e'  normalize window to give an energy of unity after overlaps
%        'wm' use Hamming window
%        'wn' use Hanning window
%        'x'  include all frames that include any of the x samples

%	   Copyright (C) Mike Brookes 1997-2014
%      Version: $Id: enframe.m 6490 2015-08-05 12:47:13Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nx=length(x(:));
if nargin<2 || isempty(win)
    win=nx;
end
if nargin<4 || isempty(m)
    m='';
end
nwin=length(win);
if nwin == 1
    lw = win;
    w = ones(1,lw);
else
    lw = nwin;
    w = win(:).';
end
if (nargin < 3) || isempty(inc)
    inc = lw;
end
nli=nx-lw+inc;
nf = max(fix(nli/inc),0);   % number of full frames
na=nli-inc*nf+(nf==0)*(lw-inc);       % number of samples left over
fx=nargin>3 && (any(m=='z') || any(m=='r')) && na>0; % need an extra row
f=zeros(nf+fx,lw);
indf= inc*(0:(nf-1)).';
inds = (1:lw);
if fx
    f(1:nf,:) = x(indf(:,ones(1,lw))+inds(ones(nf,1),:));
    if any(m=='r')
        ix=1+mod(nf*inc:nf*inc+lw-1,2*nx);
        f(nf+1,:)=x(ix+(ix>nx).*(2*nx+1-2*ix));
    else
        f(nf+1,1:nx-nf*inc)=x(1+nf*inc:nx);
    end
    nf=size(f,1);
else
    f(:) = x(indf(:,ones(1,lw))+inds(ones(nf,1),:));
end
if (nwin > 1)   % if we have a non-unity window
    f = f .* w(ones(nf,1),:);
end
if nargout>1
    if any(m=='E')
        t0=sum((1:lw).*w.^2)/sum(w.^2);
    elseif any(m=='A')
        t0=sum((1:lw).*w)/sum(w);
    else
        t0=(1+lw)/2;
    end
    t=t0+inc*(0:(nf-1)).';
end


