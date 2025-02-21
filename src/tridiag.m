function [A,H] = tridiag(A)
%TRIDIAG Returns a Hessenberg tridiagonal matrix similar to
%the input matrix, and the accumulated Householder rotations applied to it.
%   [T,H] = tridiag(A) takes a symmetric matrix A and returns
%
%   -   A tridiagonal matrix T, such that A~T (i.e. T has the same
%       eigenvalues as A);
%
%   -   The accumulated Householder rotation matrices H, such that
%       H * T * H' = A ( precision may vary, depending on cond(A) ).
        
    if ~issymmetric(A)
        error('The input matrix must be symmetric.');
    end
    
    size_A = size(A,1);
    n = size_A - 2;
    H = eye(size_A);

    for i = 1:n
        x = A(i+1:end, i);
        H_curr = Householder(x);

        % create the tridiagonal elements in the current row and column
        k_times_e1 = H_curr.k_times_e1();
        A(i+1:end, i) = k_times_e1;
        A(i, i+1:end) = k_times_e1';

        % Perform a similarity transformation on the submatrix in A
        A(i+1:end, i+1:end) = H_curr * A(i+1:end, i+1:end) * H_curr;

        % Accumulate the current Householder rotation matrix in H
        H(2:end, i+1:end) = H(2:end, i+1:end) * H_curr;
    end
end