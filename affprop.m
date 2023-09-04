function [idx a r output] = affprop(s, options)
%AFFPROP idenfities exemplar points using affinity propogation
%
% IDX = AFFPROP(S)
%   returns IDX, a m-vector of indices, such that idx(i) is the index to
%   the exemplar of the ith point. For exemple if x is an underlying data
%   matrix then x(i,:) is clustered with x(idx(i),:) and x(idx(i)) is the
%   exemplar.
%   S is a m x m similarity matrix. It need not be symmetric. The diagonal
%   indicates a prior preference that point s(i,i) is an examplar. Points
%   with higher values are more likely to be choosen
%
% IDX = AFFPROP(...,OPTIONS)
%   options is a structure that set properties for the search
%       'MaxIter', 200,...    maximum number of iterative refinements
%       'TolX',    1e-6, ...  termination tolerance for norm of message matrices
%       'StallIter', 20, ...  cumulative number of generations for which
%                             IDX does not change
%       'Dampening', .5       dampening factor (or 1-learning rate)
%       'OutputFcn',[]        function called each iteration with message
%                              matrices, fun(a,r)
%
% [IDX R A OUTPUT] = AFFRPOP(...)
%       also returns the message matrices R and A
%       R is a 'responsibility' matrix. R(i,:) is a row vector of message
%       from point i to the other points indicating how well suited point k
%       is for being an exemplar for point i. 
%       A is an availability matrix. A(:,k) is a
%       column vector of messages from point k to the other points
%       indicating how available point k is for being an exemplar. 
%       Availabilities and Responsibilities are added together to determine
%       exemplars. The value k that maximizes a(i,k) + r(i,k) indicates the
%       examplar for point i. 
%
%Example  - requires stats toolbox 
% M = [5 0; 0 5; 2.5 2.5];            % true centers
% V = [1 .9; .9 1];
% L = chol(V);
% idx = repmat( 1:3, 50, 1 );       % true assignment
% x = M(idx(:),:) + randn(150,2)*L; % data points with noise
% m = size(x,1);                    
% di = 1:(m+1):m*m;                 %index to diagonal elements
% d        = -pdist(x, 'mah');      %upper triangle similiarity
% s        = squareform(d);         %squareform 
% s(di)    = min(d(:));
% options.StallIter = 10;
% options.OutputFcn = @(a,r) affprop_plot(a,r,x,'k.');
% figure
% ex       = affprop(s, options );
%
%Example - for example that doesn't require stats toolbox ...
% See also affprop_demo 
%
%Reference
%  Frey and Dueck, "Clustering by Passing Messages Between Data Points",
%  Science 2007, 315:972-976

% Copyright 2006 Mike Boedigheimer
% Amgen Inc.
% Department of Computational Biology
% mboedigh@amgen.com


m = size(s,1);
defaultopt = struct( 'MaxIter', 100,...
                     'StallIter', 10, ...
                     'FunValCheck','off', ...
                     'Dampening', .9, ...
                     'OutputFcn',[]);
                 
if nargin < 2 
    options = [];
end;

maxiter   = optimget(options,'MaxIter',   defaultopt,'fast');
stalliter = optimget(options,'StallIter', defaultopt,'fast');
lambda    = optimget(options,'Dampening', defaultopt,'fast');
outputfcn = optimget(options,'OutputFcn', defaultopt,'fast');

if isempty(outputfcn)
    haveoutputfcn = false;
else
    haveoutputfcn = true;
end


% useful indices
ri = (1:m)';
di = 1:(m+1):m*m;         %index to diagonal elements对角元素索引

% initialize responsibility matrix初始化职责矩阵
r = zeros(m);

% availability matrix,a, evidence from other datapoints how an exemplar
% point i will make
a      = zeros(m);
a_prev = zeros(m);

%initialize exemplar vector
idx_prev = ri;

iter = 1;
converged = false;
ccount = 0;

while ~converged
    %% Part I. Update or initialize r, the responsibility matrix
    % update (or initialize) r, the responsibility matrix, such that element
    % r(i,j) indicates how good an exemple point x(j,:) would make for
    % point x(i,:)

    %  given as = a + s,
    %  set d(i,j) = max( as(i,k) ), for k not equal to j
    as = a + s;

    %rmax is the row maximum for all elements
    [rmax k] = max(as,[],2);
    kp = sub2ind( [m m], ri, k );

    %rmax2 is the row maximum for when j=k would be the row maximum
    as(kp) = nan;
    rmax2  = max(as,[],2);

    d = repmat( rmax, 1, m );
    d(kp) = rmax2;

    %     r = s - d;                                                 %eq(1)
    r = lambda*r + (1-lambda)*(s-d);       % dampended


    %% Part II. Update the availability matrix [ eq2 and eq3 ]
    %  The availability matrix is evidence that point k is an exemplar based on
    %  positive responsibiliities sent to it from the other points
    % a(i,j) = min[ 0, r(j,j) + sum( max(0, r(k,j) ) ],
    % for all k not in (i,j)

    % calculate c = max(r,0)
    c = r;
    c(c<0) = 0;     % only consider positive elements

    % calculate the sum( max(0, r(k,j) ), where k not in (i,j) this is done
    % by setting c(j,j) to zero. Then sum the columns and form a new
    % matrix,cs, by replication. Finally, subtracting c(i,j) gives
    % the desired matrix, c (reuse storage)
    c(di) = 0;                   % set diagonal elements to zero
    cs = repmat(sum(c),m,1);     % sum columns
    c = cs - c;

    %     a = min(0, repmat( diag(r)', m, 1) + c );                   %eq(2)
    %     a(di) = cs(di);                                             %eq(3)
    %
    a = lambda*a_prev + (1-lambda)*(min(0, repmat( diag(r)', m, 1) + c ));
    a(di) = lambda*a_prev(di) + (1-lambda)*cs(di);
    
    a_prev = a;

    if haveoutputfcn
        outputfcn(a,r);
    end

    iter = iter+1;
    if iter > maxiter
        converged = true;
    end

    [rmax idx] = max( a+r,[],2);
    if isequal(idx,idx_prev)
        ccount = ccount + 1;
        if ccount == stalliter
            converged = true;
        end
    else
        idx_prev = idx;
        ccount = 0;
    end
    



end

output.iter      = iter;
output.stalliter = ccount;
