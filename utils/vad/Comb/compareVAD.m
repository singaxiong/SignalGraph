function compareVAD(vad1, vad2)

plot(vad1); hold on;
plot(vad2*1.1,'r'); hold off;
legend('vad1', 'vad2'); 
ylim([0 1.2]);
