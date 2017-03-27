function A = GetDctMatrix(nFbank, nCep)

[x,y] = meshgrid(0:nFbank-1);
A = sqrt(2 / nFbank) * cos(pi * (2*x + 1) .* y / (2 * nFbank));
A(1,:) = A(1,:) / sqrt(2);
A = A(1:nCep,:);

end
