%% demonstrate affinity propagation
% 
% demonstration shows a movie while the message passaging is going on. 
% The first panel shows 2 dimensional data with a blue line connecting an
% exemplar (red point) to its constituents.  The remaining three panels
% show the messages being past. The top right shows the message sent from
% each potential exemplar (column) to the other points (rows). Brigher
% colors indicate higher availability. The lower left shows the
% responsibility matrix. This is a message sent from a point (row) to each 
% potential exemplar (column) indicating the suitability of that exemplar.
% The lower right is the combination of the two matrices and it indicates
% the current state. The column that maximizes a row is the examplar for
% that row. 
%
% Example
%    affprop_demo
% 
% See also affprop 

%% setup clustering
% much easier with stats toolbox because pdist is available

% % generate 2d data
% M = [5 0; 0 5; 10 10];
% idx = repmat( 1:3, 50, 1 );
% x = M(idx(:),:) + randn(150,2);

n_var=2;
VRmin=[1 -1];     % the low bounds of the decision variables
VRmax=[3 1];
Particle_Number=800;
VRmin=repmat(VRmin,Particle_Number,1);
VRmax=repmat(VRmax,Particle_Number,1);
x=VRmin+(VRmax-VRmin).*rand(Particle_Number,n_var);

% generate similarity matrix
m     = size(x,1);
s     = zeros(m);

o     = nchoosek(1:800,2);      % set up all possible pairwise comparisons 
xx    = x(o(:,1),:)';           % point 1
xy    = x(o(:,2),:)';           % point 2
d2    = (xx - xy).^2;           % distance squared
d     = -sqrt(sum(d2));         % distance

k     = sub2ind([m m], o(:,1), o(:,2) );    % prepare to make square 
s(k)  = d;             
s     = s + s';
di = 1:(m+1):m*m;         %index to diagonal elements

s(di) = min(d);

%% clustering
options.StallIter = 10;
options.OutputFcn = @(a,r) affprop_plot(a,r,x,'k.');

figure
ex       = affprop(s, options );
u=unique(ex );
for k = 1:length(u)
    t=ex ==u(k);
    C{k,1}=[x(t,1:n_var)];
end


