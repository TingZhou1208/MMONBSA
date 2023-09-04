function affprop_plot( a, r, x, cc )
%AFFPROP_PLOT throw away function to illustrate affinity propagation
% 
%Example
%   for example see
%See also affprop_demo, affprop

if nargin < 4
    cc = '.';
end

m = size(x,1);
subplot(2,2,1);
cla
plot(x(:,1),x(:,2), cc, 'markersize', 3 );

[rmax k] = max( a+r,[],2);


% color examplars
i = (1:m)';
hold on;
plot(x(i==k,1),x(i==k,2), 'r.', 'markersize', 7 );

j = i~=k;

% quiver( x(j,1),              x(j,2), ...
%         x(k(j),1) - x(j,1),  x(k(j),2) - x(j,2), ...
%         0 );

quiver( x(j,1),              x(j,2), ...
    x(k(j),1) - x(j,1),  x(k(j),2) - x(j,2), ...
    0, '.' );

subplot(2,2,2)
colormap(copper)
imagesc(a);
title('availability')

subplot(2,2,3);
colormap(copper)
imagesc(r);
title('responsibility')

subplot(2,2,4);
colormap(copper)
imagesc(a+r);
title('availability+responsibility');
pause(0.1);
drawnow