function [node,nodeL,nodeR] = splitNode(data,node,param)
% Split node

visualise = 0;

% Initilise child nodes
iter = param.splitNum;
nodeL = struct('idx',[],'t',nan,'dim',0,'dim2',0,'prob',[]);
nodeR = struct('idx',[],'t',nan,'dim',0,'dim2',0,'prob',[]);

if length(node.idx) <= 5 % make this node a leaf if has less than 5 data points
    node.t = nan;
    node.dim = 0;
    return;
end

idx = node.idx;
data = data(idx,:);
[N,D] = size(data);
ig_best = -inf; % Initialise best information gain
idx_best = [];
for n = 1:iter
    
    % Split function - Modify here and try other types of split function
    id = randperm(D);
    dim = id(1); % Pick one random dimension
    dim2 = id(2); % Only needed for two-pixel
    d_min = single(min(data(:,dim))) + eps; % Find the data range of this dimension
    d_max = single(max(data(:,dim))) - eps;
    switch param.splitfunc 
        case 'axis' % Axis-align
            t = d_min + rand*((d_max-d_min)); % Pick a random value within the range as threshold
            idx_ = data(:,dim) < t;
        case 'pixel' % Two-pixel
            dist = d_max - d_min;
            t = -dist + rand*(2*dist); % Pick a random value within the range as threshold
            idx_ = data(:,dim) - data(:,dim2) < t;
    end
        
    ig = getIG(data,idx_); % Calculate information gain
    
    if visualise
        visualise_splitfunc(idx_,data,dim,t,ig,n);
        pause();
    end
    
    if (sum(idx_) > 0 & sum(~idx_) > 0) % We check that children node are not empty
        [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim, dim2, idx_best);
    end
end

nodeL.idx = idx(idx_best);
nodeR.idx = idx(~idx_best);

if visualise
    visualise_splitfunc(idx_best,data,dim,t,ig_best,0)
    fprintf('Information gain = %f. \n',ig_best);
    pause();
end

end

function ig = getIG(data,idx) % Information Gain - the 'purity' of data labels in both child nodes after split. The higher the purer.
L = data(idx);
R = data(~idx);
H = getE(data);
HL = getE(L);
HR = getE(R);
ig = H - sum(idx)/length(idx)*HL - sum(~idx)/length(idx)*HR;
end

function H = getE(X) % Entropy
cdist= histc(X(:,1:end), unique(X(:,end))) + 1;
cdist= cdist/sum(cdist);
cdist= cdist .* log(cdist);
H = -sum(cdist);
end

function [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx,dim,dim2,idx_best) % Update information gain
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dim = dim;
    node.dim2 = dim2;
    idx_best = idx;
else
    idx_best = idx_best;
end
end