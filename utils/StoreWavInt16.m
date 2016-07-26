function wav = StoreWavInt16(wav)

wav = wav/max(max(abs(wav)));
wav = int16(wav*2^15);

end
