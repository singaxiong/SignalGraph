function A = GetDctMatrix(nFbank, nCep)

A = dctmtx(nFbank);

A = A(1:nCep,:);

end
