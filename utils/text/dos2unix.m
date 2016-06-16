function path_unix = dos2unix(path_dos)

for i=1:length(path_dos)
    if strcmp( path_dos(i) , '\' );
        path_unix(i) = '/';
    else
        path_unix(i) = path_dos(i);
    end
end
path_unix = regexprep(path_unix, '//', '/');
path_unix = regexprep(path_unix, '//', '/');
