function [U,s,V] = svd_custom(A,tau)
%[U,s,V] = SVD_CUSTOM(A) Custom economy SVD implementation.
%   Returns the singular values and the singular vectors of a
%   real-valued matrix A.
%   The functionality is the same as
%   [U,s,V] = svd(A, 'econ', 'vector');
%
%   The input parameters are:
%       A         - A generic real-valued matrix;
%
%       tau       - (Optional) A scalar threshold used by the
%                   QR algorithm implemented in eig_tridiag.m; see
%                   'help eig_tridiag' for details.
%
%   The function returns:
%       U   - A matrix whose columns are the left singular vectors of A;
%
%       e   - A column vector containing the singular values of A;
%
%       V   - A matrix whose columns are the right singular vectors of A.

% Set default values
if nargin < 2
    tau = 1e-8;
end

[m,n] = size(A);

if m <= n
    [U,s] = calc_singular(A * A', tau);
    V = A' * (U ./ s');
else
    [V,s] = calc_singular(A' * A, tau);
    U = A * (V ./ s');
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Helper function(s) %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X,s] = calc_singular(AA, tau)
%CALC_SINGULAR calculates
%   - the vector of singular values s of a matrix A;
%   - the matrix of singular vectors X of a matrix A
%where X are:
%   - the left singular vectors U of A, if AA = A * A';
%   - the right singular vectors V of A, if AA = A' * A.
[AA,H] = tridiag(AA);
[X,e] = eig_tridiag(AA, tau);
X = H * X;

% Sort eigenvalues and the corresponding
% eigenvectors in descending order
[e, X_sorted_indices] = sort(e,'descend');
X = X(:, X_sorted_indices);

% Compute singular values
s = sqrt(e);
end