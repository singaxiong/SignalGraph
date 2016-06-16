% Print the progress of the tasks
function PrintProgress(finished,nTask,step, tag)
noTag = 0;
if nargin<4
    noTag = 1;
end
if nargin<3
    step = 1;
end
if mod(finished, step)==0
    if noTag
        fprintf('  %d out of %d tasks (%2.1f%%) finished - %s\n', finished, nTask, finished/nTask*100, datestr(now));
    else
        fprintf('  %d out of %d tasks (%2.1f%%) finished - %s - %s\n', finished, nTask, finished/nTask*100, datestr(now), tag);
    end
end
