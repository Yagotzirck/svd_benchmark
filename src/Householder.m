classdef Householder
    %HOUSEHOLDER Pseudo - elementary Householder matrix based on input
    %vector.
    %   HOUSEHOLDER(x) takes a vector x and returns an object H s.t.
%       H * x = k * e_1
%   where:
%       - H: the Householder elementary matrix built from w, i.e.
%           H = eye(length(x) - 2*w*w');
%       - e_1: the first column of the identity matrix, having the same
%       length as x.
%   In order to reduce computations, H is not really the Householder
%   elementary matrix; instead, it's an object that exploits the rank-1 of
%   H to avoid a matrix-vector product, replacing it with a vector outer
%   product instead.
    
    properties (Access = private)
        beta
        k
        v
    end
    

    methods (Access = public)
        function obj = Householder(x)
            %HOUSEHOLDER Construct an instance of this class.
            %   H = HOUSEHOLDER(x) returns an object that simulates a
            %   Householder elementary matrix H to be used in left and
            %   right multiplications by another matrix/vector, without
            %   actually computing H.
            sigma = norm(x);

            obj.k = sigma;
            if x(1) >= 0    % k must have x(1)'s opposite sign
             obj.k = -obj.k;
            end

            obj.beta = sigma .* (sigma + abs( x(1) ) );

            % v = x - k .* e_1
            obj.v = x;
            obj.v(1) = obj.v(1) - obj.k;
        end


        function result = k_times_e1(H)
             %K_TIMES_E1 Returns the vector corresponding to the
             %right side of the equality
             %H * x = k * e_1
             %where e_1 is the first column of the identity matrix.
             result = zeros( length(H.v), 1 );
             result(1) = H.k;
        end

        
        function result = mtimes(obj1, obj2)
            %MTIMES Overload the '*' operator to simulate a matrix-matrix
            %multiplication between H and a generic matrix A, without
            %actually computing H.
            if isa(obj1, 'Householder') && isnumeric(obj2)
                result = obj1.matmul_left(obj2);    % H * A
            elseif isnumeric(obj1) && isa(obj2, 'Householder')
                result = obj2.matmul_right(obj1);   % A * H
            else
                error(['A Householder object can only be multiplied ' ...
                    'by a matrix/vector.']);
            end
        end

    end


    methods (Access = private)
        function A = matmul_left(H, A)
            %MATMUL_LEFT Returns the equivalent of H * A.
            nCols = size(A, 2);

            for j = 1:nCols
                a = A(:, j); % column vector
                gamma = (H.v' * a) ./ H.beta;
                A(:, j) = a - gamma .* H.v;
            end
        end


        function A = matmul_right(H, A)
             %MATMUL_RIGHT Returns the equivalent of A * H.
            nRows = size(A, 1);

            for i = 1:nRows
                a = A(i, :); % row vector
                gamma = (a * H.v) ./ H.beta;
                A(i, :) = a - gamma .* H.v';
            end
        end

    end
end