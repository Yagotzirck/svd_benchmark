function [U,e] = eig_tridiag(A, tau)
%EIG_TRIDIAG Calculates eigenvectors and eigenvalues of a
%symmetric tridiagonal matrix.
%   [U,e] = EIG_TRIDIAG(A, tau) computes the eigenvectors and eigenvalues
%   of a symmetric tridiagonal matrix A
%   (or equivalently, a symmetric Hessenberg matrix).
%
%   The input parameters are:
%       A   - A symmetric tridiagonal matrix;
%
%       tau - A scalar threshold used to determine when to consider 
%             subdiagonal elements as zero during each QR iteration.
%
%   The function returns:
%       U   - A matrix whose columns are the eigenvectors of A.
%             The eigenvectors corresponding to the non-zero and unique
%             eigenvalues are expected to be orthonormal; however,
%             numerical precision and A's rank may affect this.
%             Decreasing the tau parameter can improve the orthonormality
%             of the eigenvectors, but may also slow convergence; moreover,
%             a too small value can also amplify numerical errors, so a bit
%             of trial-and-error might be necessary to find the best
%             threshold.
%
%       e   - A vector containing the eigenvalues of A.
%
%   A naive spectral shift is implemented to accelerate the convergence 
%   of the QR algorithm; future improvements may include the use of 
%   more sophisticated shifts, such as the Wilkinson shift.

n = size(A,1);
U = eye(n);

while n > 1
    A_curr = A(1:n, 1:n);
    U_curr = U(1:n, 1:n);
    
    while abs( A_curr(n,n-1) ) > tau
        spectral_shift = A_curr(n,n) * diag(ones(n,1));
        [Q,R] = qr_tridiag(A_curr - spectral_shift, tau);
        A_curr = calc_R_times_Q_tridiag(R,Q) + spectral_shift;
        U_curr = U_curr * Q;
    end

    A(1:n, 1:n) = A_curr;
    U(1:n, 1:n) = U_curr;
    n = n-1;
end

e = diag(A);

% Normalize eigenvectors
eigvec_norms = vecnorm(U);
U = U ./ eigvec_norms;

% Multiply each eigenvalue by the norm of the corresponding eigenvector
e = e .* eigvec_norms';

% Sort eigenvalues and eigenvectors in descending order, based on the
% eigenvalues' updated values
[e, sorted_indices] = sort(e,'descend');
U = U(:,sorted_indices);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Helper function(s) %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = calc_R_times_Q_tridiag(R, Q)
% CALC_R_TIMES_Q Since R is tridiagonal symmetric and Q is a Hessenberg
% matrix, we can implement the product R * Q in an optimized way.
n = size(R,1);

R_diag = diag(R);
R_superdiag = diag(R,1);

Q_diag = diag(Q);
Q_subdiag = diag(Q,-1);

% Calculate diagonal elements
A_diag = R_diag .* Q_diag;
A_diag(1:n-1) = A_diag(1:n-1) + R_superdiag .* Q_subdiag;

% Calculate subdiagonal and superdiagonal elements (they're the same)
A_subdiag = R_diag(2:n) .* Q_subdiag;

% Combine the three diagonals to build the tridiagonal matrix
% for the next QR iteration
A = diag(A_diag) + diag(A_subdiag,-1) + diag(A_subdiag,1);

end