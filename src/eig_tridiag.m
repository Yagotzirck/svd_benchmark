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
%             of the eigenvectors, but may also slow convergence.
%
%       e   - A vector containing the eigenvalues of A.
%
%   Matrix deflation as well as the Raylegh quotient spectral shift are
%   implemented to accelerate the convergence of the QR algorithm;
%   future improvements may include the use of more sophisticated shifts,
%   such as the Wilkinson shift.

n = size(A,1);
U = eye(n);
e = zeros(n,1);
i = n;

while i > 1
    U_curr = eye(i);
    
    while abs( A(i,i-1) ) > tau
        spectral_shift = diag( A(i,i) * ones(i,1) );
        [Q,R] = qr_tridiag(A - spectral_shift, tau);
        A = calc_R_times_Q_tridiag(R,Q) + spectral_shift;
        U_curr = U_curr * Q;
    end

    U_updater = eye(n);
    U_updater(1:i,1:i) = U_curr;
    U = U * U_updater;  % Update the eigenvectors' matrix
    
    e(i) = A(i,i);      % Save the found eigenvalue

    i = i-1;
    A = A(1:i, 1:i);    % Deflate A
end

e(1) = A(1,1);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Helper function(s) %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = calc_R_times_Q_tridiag(R, Q)
% CALC_R_TIMES_Q Since we know that:
%   1) A = R * Q is tridiagonal symmetric (QR invariant);
%   2) R is a tridiagonal matrix (one diagonal and two superdiagonals);
%   3) Q is a Hessenberg matrix
% we can implement the product R * Q in an optimized way.
R_diag = diag(R);
R_superdiag = diag(R,1);

Q_diag = diag(Q);
Q_subdiag = diag(Q,-1);

% Calculate diagonal elements
A_diag = R_diag .* Q_diag;
A_diag(1:end-1) = A_diag(1:end-1) + R_superdiag .* Q_subdiag;

% Calculate subdiagonal and superdiagonal elements (they're the same)
A_subdiag = R_diag(2:end) .* Q_subdiag;

% Combine the three diagonals to build the tridiagonal matrix
% for the next QR iteration
A = diag(A_diag) + diag(A_subdiag,-1) + diag(A_subdiag,1);

end