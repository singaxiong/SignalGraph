function log = initializeLog(log)

if isfield(log, 'actual_LR')==0
    log.actual_LR = [];
end
if isfield(log, 'cost')==0;
    log.cost = [];
end
if isfield(log, 'cost_cv')==0;
    log.cost_cv = [];
end
if isfield(log, 'subcost')==0;
    log.subcost = [];
end
if isfield(log, 'subcost_cv')==0;
    log.subcost_cv = [];
end

end
