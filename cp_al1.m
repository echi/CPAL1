function [P,Uinit,output] = cp_al1(X, R, varargin)
%CP_AL1 Compute the l1 CP factorization of a tensor
%
%   P = CP_AL1(X, R) computes an estimate of the best rank-R
%   CP model of a tensor X using an alternating least-L1
%   algorithm.  The input X can be a tensor, sptensor, ktensor, or
%   ttensor. The result P is a ktensor.
%
%   P = CP_AL1(X, R, 'param', value, ...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%      'epsilon' - parameter to avoid divide by zero {1.0e-10}
%      'tol' - Tolerance on relative difference in fit {1.0e-5}
%      'maxiters' - Maximum number of iterations {500}
%      'dimorder' - Order to loop through dimensions {1:ndims(A)}
%      'init' - Initial guess [{'random'}|'nvecs'|cell array]
%      'printitn' - Print fit every n iterations; 0 for no printing {1}
%
%   [P,U0] = CP_AL1(...) also returns the initial guess.
%
%   [P,U0,out] = CP_ALS(...) also returns additional output that contains
%   the input parameters.
%
%
%   Examples:
%   X = sptenrand([5 4 3], 10);
%   P = cp_al1(X,2);
%   P = cp_al1(X,2,'dimorder',[3 2 1]);
%   P = cp_al1(X,2,'dimorder',[3 2 1],'init','nvecs');
%   U0 = {rand(5,2),rand(4,2),rand(3,2)}; %<-- Initial guess for factors of P
%   [P,U0,out] = cp_al1(X,2,'dimorder',[3 2 1],'init',U0);
%   P = cp_al1(X,2,out.params); %<-- Same params as previous run
%
%   Note: The "fit" is defined as 1 - norm(X-full(P))/norm(X) and is
%   loosely the proportion of the data described by the CP model, i.e., a
%   fit of 1 is perfect.
%
%   See also CP_ALS, KTENSOR, TENSOR, SPTENSOR, TTENSOR.

%% Extract dimensions of X and number of dimensions of X.
dims = size(X);
N = ndims(X);
normX = norm(X);

%% Set algorithm parameters from input or by using defaults.
params = inputParser;
params.addParamValue('epsilon',1e-10,@isscalar);
params.addParamValue('tol',1e-5,@isscalar);
params.addParamValue('dimorder',1:N,@(x) isequal(sort(x),1:N));
params.addParamValue('maxiters',500,@(x) isscalar(x) & x > 0);
params.addParamValue('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addParamValue('printitn',1,@isscalar);
params.parse(varargin{:});

%% Copy from params object.
epsilon = params.Results.epsilon;
fitchangetol = params.Results.tol;
maxiters = params.Results.maxiters;
dimorder = params.Results.dimorder;
init = params.Results.init;
printitn = params.Results.printitn;

%% Error checking 

%% Set up and error checking on initial guess for U.
if iscell(init)
    Uinit = init;
    if numel(Uinit) ~= N
        error('OPTS.init does not have %d cells',N);
    end
    
    for n = dimorder(1:end);
        if ~isequal(size(Uinit{n}),[size(X,n) R])
            error('OPTS.init{%d} is the wrong size',n);
        end
    end
else
    if strcmp(init,'random')
        Uinit = cell(N,1);
        for n = dimorder(1:end);
            Uinit{n} = rand(size(X,n),R);
        end
    elseif strcmp(init,'nvecs') || strcmp(init,'eigs') 
        Uinit = cell(N,1);
        for n = dimorder(1:end);
            Uinit{n} = nvecs(X,n,R);
        end
    else
        error('The selected initialization method is not supported');
    end
end

%% Set up for iterations - initializing U and the fit.
U = Uinit;
fit = 0;

if printitn>0
  fprintf('\nCP_AL1:\n');
end

%% Main Loop: Iterate until convergence.

for iter = 1:maxiters
    
    fitold = fit;

    
    for n = dimorder(1:end);
    %for n = 1:N

        G = ones(1,R);
        for i = [1:n-1,n+1:N]
            G = khatrirao(U{i},G);
        end
        
        Xn = double(tenmat(X,n));
        U{n} = L1RowRegression(G, Xn', U{n}', epsilon, 100, 1e-3, 1e-8)';

        % Solve for all rows simultaneously (not worth it!)
%   1. By large scale L1 magic
%        GG = sparse(kron(eye(dims(n)), G));
%        out = l1decode_pd(vec(U{n}'), @(z) GG*z, @(z) GG' *z, vec(Xn'));
%        U{n} = reshape(out, [], dims(n))';
%
%    2. By MM
%        GG = sparse(kron(eye(dims(n)), G));
%        out = L1Regv2(vec(Xn'), vec(U{n}'), GG, 'maxiters', maxiters)';
%        U{n} = reshape(out, [], dims(n))';
        
%        for i = 1:dims(n)       
%          --- 1. MM
%            U{n}(i,:) = L1Reg(Xn(i,:)', U{n}(i,:)', G, 'maxiters', maxiters, 'tol', 1e-3);
%            U{n}(i,:) = L1Regression(G, Xn(i,:)', U{n}(i,:)', epsilon, 100, 1e-3, 1e-10);
%            U{n}(i,:) = L1RowRegression(G, Xn(i,:)', U{n}(i,:)', epsilon, 100, 1e-3, 1e-10);
%
%          --- 2. CVX LP (second slowest!)
%            cvx_begin quiet
%                variable z(R);
%                minimize( norm(Xn(i,:)' - G * z,1) );
%            cvx_end
%            U{n}(i,:) = z;
%
%          --- 3. Small scale L1 magic
%            U{n}(i,:) = l1decode_pd(U{n}(i,:)', G, [], Xn(i,:)', 1e-4, 30);
%
%          --- 4. Large sale L1 magic
%            U{n}(i,:) = l1decode_pd(U{n}(i,:)', @(z) G*z, @(z) G'*z, Xn(i,:)');
%
%          --- 5. linprog (slowest!)
%             [Gm, Gn] = size(G);
%             f    = [ zeros(Gn,1); ones(Gm,1);  ones(Gm,1)  ];
%             Geq  = [G,            -eye(Gm),    +eye(Gm)    ];
%             lb   = [ -Inf(Gn,1);  zeros(Gm,1); zeros(Gm,1) ];
%             xzz  = linprog(f,[],[],Geq,Xn(i,:)',lb,[]);
%             U{n}(i,:) = xzz(1:Gn,:);
%
%          --- 6. CLP (slow)
%            [Gm, Gn] = size(G);
%            c    = [ zeros(Gn,1); ones(Gm,1);  ones(Gm,1)  ];
%            Geq  = [G,            -eye(Gm),    +eye(Gm)    ];
%            lb   = [ -Inf(Gn,1);  zeros(Gm,1); zeros(Gm,1) ];
%            xzz  = clp([], c, [], [], Geq, Xn(i,:)',lb,[]);
%            U{n}(i,:) = xzz(1:Gn,:);  
%        end

        for i = 1:R
            lambda(i) = norm(U{n}(:,i));
            if (lambda(i) > 0)
                U{n}(:,i) = (1/lambda(i))*U{n}(:,i);
            end
        end
    end

    P = ktensor(lambda',U);
    normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) );
    fit = 1 - (normresidual / normX); %fraction explained by model
    fitchange = abs(fitold - fit);
    
%    fit = sum(sqrt(double(tenmat(tensor(X) - tensor(P), 1:N, [])).^2 + epsilon));
%    fitchange = abs(fitold - fit)/(abs(fitold) + 1);
    
    if mod(iter,printitn)==0
      fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n', iter, fit, fitchange);
    end

    % Check for convergence
    if (iter > 1) && (fitchange < fitchangetol)
        break;
    end

end

%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
P = arrange(P);
% Fix the signs
P = fixsigns(P);

if printitn>0
  normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) );
  fit = 1 - (normresidual / normX); %fraction explained by model
  fprintf(' Final fit = %e \n', fit);
end

output = struct;
output.params = params.Results;
