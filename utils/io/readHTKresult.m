function readHTKresult(input, output);

INPUT = fopen(input,'r');
OUTPUT = fopen(output,'w');

% Read in the data for one test
for i=1:70
    fsearch2('Date:',INPUT);
    tmp = textscan(INPUT,'%s%s%d%s%d',1);
    data.date.week(i) = tmp{1};
    data.date.month(i) = tmp{2};
    data.date.day(i) = tmp{3};
    data.date.time(i) = tmp{4};
    data.date.year(i) = tmp{5};
    
    fsearch2('labels/',INPUT);
    tmp = textscan(INPUT,'%2c',1);
    data.label(i) = tmp;
    
    fsearch2('Correct=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.sent_corr(i) = tmp{1};
    fsearch2('H=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.sent_H(i) = tmp{1};
    fsearch2('S=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.sent_S(i) = tmp{1};
    fsearch2('N=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.sent_N(i) = tmp{1};
    
    fsearch2('Corr=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.word_corr(i) = tmp{1};
    fsearch2('Acc=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.word_acc(i) = tmp{1};
    fsearch2('H=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.word_H(i) = tmp{1};
    fsearch2('D=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.word_D(i) = tmp{1};
    fsearch2('S=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.word_S(i) = tmp{1};
    fsearch2('I=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.word_I(i) = tmp{1};
    fsearch2('N=',INPUT);
    tmp = textscan(INPUT,'%f',1);
    data.word_N(i) = tmp{1};
end

% write the data
for i=1:70
    if i==1, fprintf(OUTPUT,'\nTEST SET A\n');
    elseif i==29, fprintf(OUTPUT,'\nTEST SET B\n');
    elseif i==57, fprintf(OUTPUT,'\nTEST SET C\n'); end
    if mod(i,7) == 1, fprintf(OUTPUT,'%s\n',data.label{i}); end
%     fprintf(OUTPUT,'%s %s %d %s %d ',...
%         data.date.week{i},data.date.month{i},data.date.day(i),data.date.time{i},data.date.year(i));
    fprintf(OUTPUT,'%s %d %s - ',data.date.month{i},data.date.day(i),data.date.time{i});
    fprintf(OUTPUT,'%2.2f	%2.2f   	%d	%d   	%d   	%d  	%d  - ',data.word_corr(i),data.word_acc(i),data.word_H(i),...
        data.word_D(i),data.word_S(i),data.word_I(i),data.word_N(i));
    fprintf(OUTPUT,'%2.2f %d %d %d \n',data.sent_corr(i),data.sent_H(i),data.sent_S(i),data.sent_N(i));
end
fprintf(OUTPUT,'\n');

fclose(INPUT);
fclose(OUTPUT);