op = [1 0.357 0.8533 0.028; 0.9600 0.241 0.6467 0.050; 0.9800 0.974 0.6533 0.024;  0.9200 3.553 0.4933 0.105]';
in = 5*[1 10 100 1000];
figure();
yyaxis left
hold on
plot(in, 100*op(1,:), "Marker","o", "LineStyle",":")
plot(in, 100*op(3,:), 'Marker',"hexagram", "LineStyle","-")
xlabel('k (log scale)')
ylabel('Accuracy (%s)')
set(gca, 'XScale', 'log');
legend('Train','Test',"Location","northwest")
yyaxis right
plot(in, op(2,:), "Marker","o", "LineStyle",":")
plot(in, op(4,:), 'Marker',"hexagram", "LineStyle","-")
ylabel('Time(s)')
legend('Train','Test')
hold off