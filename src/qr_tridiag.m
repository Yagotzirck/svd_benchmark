function [Q, A] = qr_tridiag(A, tau)
%QR_TRIDIAG Returns a QR factorization of a tridiagonal matrix.
%   [Q, A] = QR_TRIDIAG(A, tau) computes the QR factorization of a 
%   tridiagonal square matrix A using Givens rotations.  The function 
%   modifies A in place and returns the orthogonal matrix Q and the 
%   upper triangular matrix A.
%
%   Inputs:
%       A   - A square tridiagonal matrix of size n-by-n.
%       tau - A scalar threshold for determining which subdiagonal 
%             elements to zero out. Elements with absolute values 
%             greater than tau will be targeted.
%
%   Outputs:
%       Q   - An orthogonal matrix such that Q'*Q = I.
%       A   - An upper triangular matrix corresponding to R.
%
%   Note: This function does not check if A is tridiagonal. Ensure that 
%   the input matrix A is tridiagonal to avoid unexpected results.
%
%   Example:
%       A = [4, 1, 0; 1, 4, 1; 0, 1, 4];
%       tau = 0.1;
%       [Q, R] = qr_tridiag(A, tau);
%
%   This example demonstrates how to apply the QR factorization to a 
%   tridiagonal matrix A with a specified threshold tau.
n = size(A,1);
if n ~= size(A,2) || n < 2
    error('The input matrix must be square and at least 2x2.');
end

Q = eye(n);

for k = 1:n-2
    [Q,A] = qr_step(A, Q, tau, k, 3);
end

% Last iteration
k = n-1;
[Q,A] = qr_step(A, Q, tau, k, 2);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Helper function(s) %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [c,s] = calc_givens_coeffs(x_i, x_j)
%CALC_GIVENS_COEFFS Given two scalars x_i, x_j corresponding to the
%elements x(i) and x(j) of a vector x, it returns the cosine and sine
%parameters to build a Givens rotation matrix G, such that
%y = G * x results in a vector having the element y(j) = 0.
if x_i >= x_j
    t = x_j / x_i;
    c = 1 / sqrt(1 + t^2);
    s = t * c;
else
    t = x_i / x_j;
    s = 1 / sqrt(1 + t^2);
    c = t * s;
end

end


function [Q,A] = qr_step(A, Q, tau, k, submatrix_cols)
x_j = A(k+1,k);
if abs(x_j) > tau
    x_i = A(k,k);
    [c,s] = calc_givens_coeffs(x_i, x_j);
    G = [c,s; -s,c];
    
    % In a tridiagonal matrix, we have the following situation,
    % up to k <= n-2:
    %
    % 1) the diagonal element A(k,k) has only one non-zero element to
    % its right;
    %
    % 2) the subdiagonal element A(k+1,k) has only two non-zero
    % elements to its right.
    %
    % Therefore, we can avoid useless computations by operating on the
    % submatrix A(k:k+1, k:k+2), instead of the whole rows A(k,:) and
    % A(k+1,:).
    %
    % As for the last iteration (k = n-1), both the diagonal and
    % subdiagonal element have only one non-zero element to their
    % right; this case it will be treated outside of the main for loop.
    A(k:k+1,k:k+(submatrix_cols-1)) = G * A(k:k+1, k:k+(submatrix_cols-1));
    A(k+1,k) = 0;   % Remove rounding errors from the subdiag element

    % Accumulate the current Givens rotation matrix in Q.
    % Note that only the columns k and k+1 are affected, up to the
    % (k+1)-th row; therefore, we can avoid useless computations in here
    % as well (for example, by keeping G in its compact form 2x2, instead
    % of embedding it inside an identity matrix n x n and performing a
    % full n x n matrix multiplication Q * G').
    Q(1:k+1, k:k+1) = Q(1:k+1, k:k+1) * G';
end

end