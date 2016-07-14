function wavlen = DecideWavLen4XFrames(nFr, frame_len, frame_shift)
    wavlen = (nFr-1)*frame_shift + frame_len;
end