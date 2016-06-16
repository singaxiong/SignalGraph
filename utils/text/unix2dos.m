function path_dos = unix2dos(path_unix)

for i=1:length(path_unix)
    if strcmp( path_unix(i) , '/' );
        path_dos(i) = '\';
    else
        path_dos(i) = path_unix(i);
    end
end

path_dos = regexprep(path_dos, '\\', '\');
path_dos = regexprep(path_dos, '\\', '\');