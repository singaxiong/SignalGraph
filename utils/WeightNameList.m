% return the field name that can contain tunable weights 
function list = WeightNameList(type)

switch lower(type)
    case 'tunable'
        list = {'W', 'b'};
    case 'all'
        list = {'W', 'b', 'prior', 'mu', 'invCov', 'mask'};
end

end