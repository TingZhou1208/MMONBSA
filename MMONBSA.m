function [ps,pf]=MMONBSA(func_name,VRmin,VRmax,n_obj,Particle_Number,Max_Gen)

% MO_Ring_PSO_SCD: A multi-objective particle swarm optimization using ring topology for solving multimodal multi-objective optimization problems
% Dimension: n_var --- dimensions of decision space
%            n_obj --- dimensions of objective space
%% Input:
%                      Dimension                    Description
%      func_name       1 x length function name     the name of test function
%      VRmin           1 x n_var                    low bound of decision variable
%      VRmax           1 x n_var                    up bound of decision variable
%      n_obj           1 x 1                        dimensions of objective space
%      Particle_Number 1 x 1                        population size
%      Max_Gen         1 x 1                        maximum  generations

%% Output:
%                     Description
%      ps             Pareto set
%      pf             Pareto front
%%  Reference and Contact
% Reference: [1]Caitong Yue, Boyang Qu and Jing Liang, "A Multi-objective Particle Swarm Optimizer Using Ring Topology for Solving Multimodal Multi-objective Problems",  IEEE Transactions on Evolutionary Computation, 2017, DOI 10.1109/TEVC.2017.2754271.
%            [2]Jing Liang, Caitong Yue, and Boyang Qu, “ Multimodal multi-objective optimization: A preliminary study”, IEEE Congress on Evolutionary Computation 2016, pp. 2454-2461, 2016.
% Contact: For any questions, please feel free to send email to zzuyuecaitong@163.com.

%% Initialize parameters
n_var=size(VRmin,2);               %Obtain the dimensions of decision space
Max_FES=Max_Gen*Particle_Number;   %Maximum fitness evaluations
%% Initialize particles' positions and velocities
VRmin=repmat(VRmin,Particle_Number,1);
VRmax=repmat(VRmax,Particle_Number,1);
pos=VRmin+(VRmax-VRmin).*rand(Particle_Number,n_var); %initialize the positions of the particles
oldpos=VRmin+(VRmax-VRmin).*rand(Particle_Number,n_var);
%% Evaluate the population
fitness=zeros(Particle_Number,n_obj);
for ii=1:Particle_Number
    fitness(ii,:)=feval(func_name,pos(ii,:));
end
fitcount=Particle_Number;            % count the number of fitness evaluations
particle=[pos,fitness];              %put positions and velocities in one matrix
EXA=[];

for gen=1:Max_Gen
    C=[];
    CC=[];
    %% 种群聚类
    % generate similarity matri
    x=pos;
    m     = size(x,1);
    s     = zeros(m);
    
    o     = nchoosek(1:Particle_Number,2);      % set up all possible pairwise comparisons
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
    % options.OutputFcn = @(a,r) affprop_plot(a,r,x,'k.');
    %
    % figure
    ex = affprop(s, options );
    u=unique(ex );
    for k = 1:length(u)
        t=ex ==u(k);
        C{k,1}=x(t,1:n_var);
        CC{k,1}=oldpos(t,1:n_var);
    end
    
    particle_position=C;
    old_position=CC;
    num=length(particle_position);
    %% 在每个子种群中进化
    for ij=1:num
        subsize=size(particle_position{ij,1},1);
        for jj=1:subsize
            for h=1:Particle_Number
                if isequal(particle_position{ij,1}(jj,1:n_var),particle(h,1:n_var))
                    particle_position{ij,1}(jj,n_var+1:n_var+n_obj)=particle(h,n_var+1:n_var+n_obj);
                    particle(h,1:n_var)=inf;
                    break;
                end
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ij=1:num
        subsize=size(particle_position{ij,1},1);
        n_GBA=2*subsize;
        fitness_T=[];
        particle_T=[];
        
        temp_particle=non_domination_scd_sort(particle_position{ij,1}, n_obj, n_var);
        tempindex=find(temp_particle(:,n_var+n_obj+1)==1);
        GBA{ij,1}=temp_particle(tempindex,1:n_var+n_obj);
        subgbest{ij,1}=GBA{ij,1}(1,1:n_var+n_obj);
        %% 选择I
        if rand(1) < rand(1)
            old_position{ij,1}= particle_position{ij,1}(:,1:n_var);                  %把P的值赋给oldP
        end
        index = randperm(subsize);
        old_position{ij,1}= old_position{ij,1}(index,:);          %打乱oldP种群个体位置 确保BSA指定的一个种群属于随机选择的前一代作为历史种群
        %% 变异
        F=3*randn(1);
        M = particle_position{ij,1}(:,1:n_var) + F.*(old_position{ij,1}(:,1:n_var)-particle_position{ij,1}(:,1:n_var));
        %% 交叉（两种交叉策略 + 交叉过程）
        map=ones(subsize ,n_var);              %二进制整数值矩阵（映射）
        mixrate=1;                             % 交叉概率
        if rand(1)<rand(1)                     % 策略1 --- 多维交叉
            for k=1:subsize
                u=randperm(n_var);             % 打乱维度
                map(k,u(1:ceil(mixrate*rand(1)*n_var)))=0;
            end
        else                                   % 策略2 --- 一维交叉
            for k=1:subsize
                map(k,randi(n_var))=0;
            end
        end
        T=M;                                        % 变异种群M作为初始的试验种群T
        for jj=1:subsize
            for j=1:n_var
                if map(jj,j)==1
                    T(jj,j)=particle_position{ij,1}(jj,j);          % P种群交叉了M中部分个体，得到T;
                end
            end
            %% 局部搜索4
            gamma=0.9;
            sigma=1-10.^(-n_var.*gen./Max_Gen);
            r= rand(1) * ( 1 - exp( -gamma * gen ) );
            if rand>r
                for j=1:n_var
                   T(jj,j)=subgbest{ij,1}(1,j)+normrnd(particle_position{ij,1}(jj,j),sigma^2); %%小生境中个体
                end
            end         
            %% 边界处理
            for j=1:n_var
                if T(jj,j) < VRmin(1,j)||T(jj,j) > VRmax(1,j)
                    T(jj,j)=VRmin(1,j)+(VRmax(1,j)-VRmin(1,j))*rand(1);
                end
            end
            %% Evaluate the population
            fitness_T(jj,:)=feval(func_name,T(jj,1:n_var));
            fitcount=fitcount+1;
            particle_T(jj,1:n_var+n_obj)=[T(jj,:),fitness_T(jj,:)];
        end
        %% 更新子种群
        temp=[particle_T(:,1:n_var+n_obj);particle_position{ij,1}(:,1:n_var+n_obj)];
        temp=unique(temp,'rows','stable');
        temp=non_domination_scd_sort(temp(:,1:n_var+n_obj), n_obj, n_var);
        particle_position{ij,1}=temp(1:subsize,1:n_var+n_obj);
        %% Sort the non-dominated particles as rep
        firstfront_all=length(find(temp(:,n_var+n_obj+1)==1));
        rep1=temp(1:firstfront_all,:);
        %% Update GBA
        tempGBA=[GBA{ij,1}(:,1:n_var+n_obj);rep1(:,1:n_var+n_obj)];
        tempGBA=unique(tempGBA,'rows','stable');
        tempGBA=non_domination_scd_sort(tempGBA(:,1:n_var+n_obj), n_obj, n_var);
        if size(tempGBA,1)>n_GBA
            GBA{ij,1}=tempGBA(1:n_GBA,1:n_var+n_obj);
        else
            GBA{ij,1}=tempGBA(:,1:n_var+n_obj);
        end
    end
    %% Update 合并子种群
    particle=cell2mat(particle_position);
    oldpos=cell2mat(old_position);
    pos=particle(:,1:n_var);
    %% Update EXA
    tempEXA=cell2mat(GBA);
    tempEXA=[EXA;tempEXA(:,1:n_var+n_obj)];
    tempEXA=unique(tempEXA,'rows','stable');
    tempEXA=non_domination_scd_sort(tempEXA(:,1:n_var+n_obj), n_obj, n_var);
    if size(tempEXA,1)>Particle_Number
        EXA=tempEXA(1:Particle_Number,1:n_var+n_obj);
    else
        EXA=tempEXA(:,1:n_var+n_obj);
    end
    %     gen
    %     clf;
    %     figure(1)
    %     plot(particle(:,1),particle(:,2),'r+')
    %     pause(0.01)
    if fitcount>Max_FES
        break;
    end
end
%% Output ps and pf
tempEXA=EXA;
tempEXA=non_domination_scd_sort(tempEXA(:,1:n_var+n_obj), n_obj, n_var);
if size(tempEXA,1)>Particle_Number
    EXA=tempEXA(1:Particle_Number,:);
else
    EXA=tempEXA;
end
tempindex=find(EXA(:,n_var+n_obj+1)==1);% Find the index of the first rank particles
ps=EXA(tempindex,1:n_var);
pf=EXA(tempindex,n_var+1:n_var+n_obj);
end

function f = non_domination_scd_sort(x, n_obj, n_var)
% non_domination_scd_sort:  sort the population according to non-dominated relationship and special crowding distance
%% Input：
%                      Dimension                      Description
%      x               num_particle x n_var+n_obj     population to be sorted
%      n_obj           1 x 1                          dimensions of objective space
%      n_var           1 x 1                          dimensions of decision space

%% Output:
%              Dimension                                  Description
%      f       N_particle x (n_var+n_obj+4)               Sorted population
%    in f      the (n_var+n_obj+1)_th column stores the front number
%              the (n_var+n_obj+2)_th column stores the special crowding distance
%              the (n_var+n_obj+3)_th column stores the crowding distance in decision space
%              the (n_var+n_obj+4)_th column stores the crowding distance in objective space
global indexChange isInputParticle;
for a=1:size(x,1)
    if ~indexChange
        x(a,n_obj + n_var+n_obj + n_var+1+1)=a;
        if a==size(x,1)
            indexChange=1;
        end
    else
        if isInputParticle
            x(a,n_obj + n_var+n_obj + n_var+1+1)= x(a,n_obj + n_var+4+1);
        end
    end
end

[N_particle, ~] = size(x);% Obtain the number of particles

% Initialize the front number to 1.
front = 1;

% There is nothing to this assignment, used only to manipulate easily in
% MATLAB.
F(front).f = [];
individual = [];

%% Non-Dominated sort.


for i = 1 : N_particle
    % Number of individuals that dominate this individual
    individual(i).n = 0;
    % Individuals which this individual dominate
    individual(i).p = [];
    for j = 1 : N_particle
        dom_less = 0;
        dom_equal = 0;
        dom_more = 0;
        for k = 1 : n_obj
            if (x(i,n_var + k) < x(j,n_var + k))
                dom_less = dom_less + 1;
            elseif (x(i,n_var + k) == x(j,n_var + k))
                dom_equal = dom_equal + 1;
            else
                dom_more = dom_more + 1;
            end
        end
        if dom_less == 0 && dom_equal ~= n_obj
            individual(i).n = individual(i).n + 1;
        elseif dom_more == 0 && dom_equal ~= n_obj
            individual(i).p = [individual(i).p j];
        end
    end
    if individual(i).n == 0
        x(i,n_obj + n_var + 1) = 1;
        F(front).f = [F(front).f i];
    end
end
% Find the subsequent fronts
while ~isempty(F(front).f)
    Q = [];
    for i = 1 : length(F(front).f)
        if ~isempty(individual(F(front).f(i)).p)
            for j = 1 : length(individual(F(front).f(i)).p)
                individual(individual(F(front).f(i)).p(j)).n = ...
                    individual(individual(F(front).f(i)).p(j)).n - 1;
                if individual(individual(F(front).f(i)).p(j)).n == 0
                    x(individual(F(front).f(i)).p(j),n_obj + n_var + 1) = ...
                        front + 1;
                    Q = [Q individual(F(front).f(i)).p(j)];
                end
            end
        end
    end
    front =  front + 1;
    F(front).f = Q;
end
% Sort the population according to the front number
[~,index_of_fronts] = sort(x(:,n_obj + n_var + 1));
for i = 1 : length(index_of_fronts)
    sorted_based_on_front(i,:) = x(index_of_fronts(i),:);
end
current_index = 0;

%% SCD. Special Crowding Distance

for front = 1 : (length(F) - 1)
    
    crowd_dist_obj = 0;
    y = [];
    previous_index = current_index + 1;
    for i = 1 : length(F(front).f)
        y(i,:) = sorted_based_on_front(current_index + i,:);%put the front_th rank into y
    end
    current_index = current_index + i;
    % Sort each individual based on the objective
    sorted_based_on_objective = [];
    for i = 1 : n_obj+n_var
        [sorted_based_on_objective, index_of_objectives] = ...
            sort(y(:,i));
        sorted_based_on_objective = [];
        for j = 1 : length(index_of_objectives)
            sorted_based_on_objective(j,:) = y(index_of_objectives(j),:);
            r=0.5;        end
        f_max = ...
            sorted_based_on_objective(length(index_of_objectives), i);
        f_min = sorted_based_on_objective(1,  i);
        
        if length(index_of_objectives)==1
            y(index_of_objectives(1),n_obj + n_var + 1 + i) = 1;  %If there is only one point in current front
        elseif i>n_var
            % deal with boundary points in objective space
            % In minimization problem, set the largest distance to the low boundary points and the smallest distance to the up boundary points
            y(index_of_objectives(1),n_obj + n_var + 1 + i) = 1;
            y(index_of_objectives(length(index_of_objectives)),n_obj + n_var + 1 + i)=0;
        else
            % deal with boundary points in decision space
            % twice the distance between the boundary points and its nearest neibohood
            y(index_of_objectives(length(index_of_objectives)),n_obj + n_var + 1 + i)...
                = 2*(sorted_based_on_objective(length(index_of_objectives), i)-...
                sorted_based_on_objective(length(index_of_objectives) -1, i))/(f_max - f_min);
            y(index_of_objectives(1),n_obj + n_var + 1 + i)=2*(sorted_based_on_objective(2, i)-...
                sorted_based_on_objective(1, i))/(f_max - f_min);
        end
        for j = 2 : length(index_of_objectives) - 1
            next_obj  = sorted_based_on_objective(j + 1, i);
            previous_obj  = sorted_based_on_objective(j - 1,i);
            if (f_max - f_min == 0)
                y(index_of_objectives(j),n_obj + n_var + 1 + i) = 1;
            else
                y(index_of_objectives(j),n_obj + n_var + 1 + i) = ...
                    (next_obj - previous_obj)/(f_max - f_min);
            end
        end
    end
    %% Calculate distance in decision space
    crowd_dist_var = [];
    crowd_dist_var(:,1) = zeros(length(F(front).f),1);
    for i = 1 : n_var
        crowd_dist_var(:,1) = crowd_dist_var(:,1) + y(:,n_obj + n_var + 1 + i);
    end
    crowd_dist_var=crowd_dist_var./n_var;
    avg_crowd_dist_var=mean(crowd_dist_var);
    %% Calculate distance in objective space
    crowd_dist_obj = [];
    crowd_dist_obj(:,1) = zeros(length(F(front).f),1);
    for i = 1 : n_obj
        crowd_dist_obj(:,1) = crowd_dist_obj(:,1) + y(:,n_obj + n_var + 1+n_var + i);
    end
    crowd_dist_obj=crowd_dist_obj./n_obj;
    avg_crowd_dist_obj=mean(crowd_dist_obj);
    %% Calculate special crowding distance
    special_crowd_dist=zeros(length(F(front).f),1);
    for i = 1 : length(F(front).f)
        if crowd_dist_obj(i)>avg_crowd_dist_obj||crowd_dist_var(i)>avg_crowd_dist_var
            special_crowd_dist(i)=max(crowd_dist_obj(i),crowd_dist_var(i)); % Eq. (6) in the paper
        else
            special_crowd_dist(i)=min(crowd_dist_obj(i),crowd_dist_var(i)); % Eq. (7) in the paper
        end
    end
    y(:,n_obj + n_var + 2) = special_crowd_dist;
    y(:,n_obj+n_var+3)=crowd_dist_var;
    y(:,n_obj+n_var+4)=crowd_dist_obj;
    if isInputParticle
        y(:,n_obj+n_var+4+1)= y(:,n_obj + n_var+n_obj + n_var+1+1);
    end
    [~,index_sorted_based_crowddist]=sort(special_crowd_dist,'descend');%sort the particles in the same front according to SCD
    y=y(index_sorted_based_crowddist,:);
    if   isInputParticle
        y = y(:,1 : n_obj + n_var+4+1 );
    else
        y = y(:,1 : n_obj + n_var+4 );
    end
    z(previous_index:current_index,:) = y;
end

f = z();
end



%Write by Caitong Yue 2017.09.04
%Supervised by Jing Liang and Boyang Qu