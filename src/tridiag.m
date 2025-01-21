function A = tridiag(A)
%TRIDIAG Returns a Hessenberg tridiagonal matrix similar to
%the input matrix.
%   H = tridiag(A) takes a symmetric matrix A and returns a tridiagonal
%   matrix H, such that A~H (i.e. H has the same eigenvalues as A).
    if ~issymmetric(A)
        error('The input matrix must be symmetric.');
    end

    n = size(A,1) - 2;

    for i = 1:n
        x = A(i+1:end, i);
        H = Householder(x);

        % create the tridiagonal elements in the current row and column
        k_times_e1 = H.k_times_e1();
        A(i+1:end, i) = k_times_e1;
        A(i, i+1:end) = k_times_e1';

        % Perform a similarity transformation on the submatrix in A
        A(i+1:end, i+1:end) = H * A(i+1:end, i+1:end) * H;
    end
end