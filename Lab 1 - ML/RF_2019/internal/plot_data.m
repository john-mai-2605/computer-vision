function  plot_data( data )
for c = 1:10
    plot(data(data(:,end)==c,1), data(data(:,end)==c,2), 'o', 'MarkerFaceColor', [c/10, 0.5+c/20, 1-c/10], 'MarkerEdgeColor','k');
    hold on;
axis([0 30 -20 20]);
end

