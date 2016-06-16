
ryear = version('-release');
ryear = str2num(ryear(1:end-1));
if ryear>2013
  set(groot,'DefaultFigureColormap',jet)
end

lib_dir = pwd;
path(path,lib_dir);

path(path,genpath([lib_dir '/utils']));
path(path,genpath([lib_dir '/graph']));
path(path,genpath([lib_dir '/signal']));
path(path,genpath([lib_dir '/prototypes']));
path(path,genpath([lib_dir '/tools']));
