function [index, f1] = trackSelection(tracks)
%% Select the trajectory bt clicking close to one end and pressing enter

%% Plot Tracks
f1 = figure;
distance = nan(size(tracks,1),1);
for ii=1:size(tracks,1)
    plot(tracks{ii,1}(:,2),tracks{ii,1}(:,3), 'linewidth', 1);
    hold all;
    title('Select tracks by clicking at one end and pressing enter in the end ');
    xlabel('x');
    ylabel('y');

end
axis equal
set(gca,'ydir','reverse');

[xs,ys] = getpts;   

index = nan(size(xs,1),1);
for jj=1:size(xs,1)
    for ii = 1:size(tracks,1)
        distance(ii,1) = sqrt((ys(jj)-tracks{ii}(end,3))^2+...
            (xs(jj)-tracks{ii}(end,2))^2);                                 %distance from the end
        distance(ii,2) = sqrt((ys(jj)-tracks{ii}(1,3))^2+...
            (xs(jj)-tracks{ii}(1,2))^2);                                   %distance from the beggining
    end
    [index(jj,1),~] = find(distance==min(distance(:)));
end

%% Replot Tracks to Confirm Selection
for ii=1:length(index)
    plot(tracks{index(ii),1}(:,2),tracks{index(ii),1}(:,3),'k','linewidth',2);              
    hold all;
end
pause(1.5)
