function feat = InputReader(files,reader, batch_mode, useGPU, precision)
if nargin<5
    precision = 'single';
end
if nargin<4
    useGPU = 0;
end
if nargin<3
    batch_mode = 0;
end

if batch_mode == 0
    files2{1} = files;
    files = files2;
end

switch lower(reader.name)
    case 'wavfile'
        [feat] = Reader_waveform(files, reader);
    case 'htk'
        if isfield(reader, 'big_endian')
            feat = Reader_HTK(files, reader.big_endian, precision);
        else
            feat = Reader_HTK(files, 0);
        end            
        if isfield(reader, 'transpose') && reader.transpose
            for i=1:length(feat)
                feat{i} = feat{i}';
            end
        end
        
    case 'wavfile2spectrum'
        feat = Reader_wavfile2spectrum(files, reader, useGPU, precision);
        
    case 'htkfile2spectrum'
        feat = Reader_htkfile2spectrum(files, reader.big_endian, useGPU, precision);
        
    otherwise
        fprintf('Unkonwn reader type\n');
        
end
if batch_mode ==0
    feat = feat{1};
end

end
