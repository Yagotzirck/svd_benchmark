classdef BenchmarkSVD
    %BENCHMARKSVD Executes SVD-related performance and accuracy tests.
    %   BENCHMARKSVD(matr_type, m_interval, n_interval, tau, sigmas)
    %   performs a series of economy SVD decompositions, comparing
    %   performance and accuracy between the built-in MATLAB svd()
    %   implementation and the custom algorithm implemented in svd_custom().
    %
    %   The input parameters are:
    %       matr_type   - A string indicating the type of test matrix to
    %                     build, on which SVD decomp. will be performed.
    %                     The list of valid values is:
    %                     'hilb', 'pascal', 'orth', 'custom sigmas'.
    %                     The first 3 values are self-explanatory; as for
    %                     'custom sigmas', it builds random singular
    %                     vectors U and V, and builds the test matrix from
    %                     them using the singular values specified in the
    %                     vector parameter 'sigmas';
    %
    %       m_list      - A vector containing the number of rows for each
    %                     test matrix;
    %
    %       n_list      - A vector containing the number of columns for
    %                     each test matrix;
    %
    %       tau         - (Optional) A scalar threshold used by the
    %                     QR algorithm implemented in eig_tridiag.m (see
    %                     'help eig_tridiag' for details);
    %
    %       sigmas      - A cell array where each element i contains the
    %                     singular values to be used when building
    %                     the i-th test matrix; this parameter is used only
    %                     if matrix_type == 'custom sigmas', and should be
    %                     ignored / left blank otherwise.
    %
    %
    %   The returned object contains the following properties:
    %
    %       type    - A string indicating the matrix type on which the
    %                 tests have been performed.
    %                 The list of possible values is:
    %                 'Hilbert', 'Pascal', 'orthogonal', 'custom sigmas'.
    %
    %       s_acc   - A cell matrix of two columns, where the first column
    %                 contains a cell array of the relative errors of the
    %                 singular values obtained by calling MATLAB's svd()
    %                 on each test matrix, whereas the second column
    %                 contains the relative errors of the singular values
    %                 obtained by calling svd_custom().
    %                 Each vector is calculated as
    %                 abs( (sigmas - sigmas_calc) ./ sigmas );
    %                 the closer the values are to 0, the better.
    %                 NOTE: This property is initialized only if
    %                 matr_type == 'custom sigmas', for obvious reasons.
    %       
    %       U_acc   - A matrix of two columns, where the first column
    %                 contains the accuracies of the left singular vectors
    %                 obtained by calling MATLAB's svd() on each test
    %                 matrix, whereas the second column contains the same
    %                 accuracy measurements on the left singular vectors
    %                 obtained by calling svd_custom().
    %                 Each value is calculated as the infinity norm of
    %                 U'U - I; the closer the values are to 0, the better.
    %
    %       V_acc   - Same as U_acc, but related to the
    %                 right singular vectors.
    %
    %       times   - A matrix of two columns, where the first column
    %                 contains the time taken to compute the SVD
    %                 factorization using MATLAB's svd() for each test
    %                 matrix, whereas the second column contains the same,
    %                 but related to svd_custom().
    
    properties (Access = public)
        type
        s_acc   % Initialized only if matr_type == 'custom sigmas'
        U_acc
        V_acc
        times
    end

    methods (Access = public)
        function obj = BenchmarkSVD(matr_type, m_list, n_list, tau, sigmas)
            %BENCHMARKSVD Construct an instance of this class.
            %   Type 'help BenchmarkSVD' for details.
            if length(m_list) ~= length(n_list)
                error(...
                    "The vector parameters 'm_list' and 'n_list' " + ...
                    "must have the same length."...
                );
            end

            if nargin == 5 && length(sigmas) ~= length(m_list)
                error( ...
                    "The cell parameter 'sigmas' must have as " + ...
                    "many columns as the length of " + ...
                    "vectors 'm_list' and 'n_list'."...
                    );
            end
            
            %% Set default values
            if nargin < 4
                tau = 1e-8;
            end

            num_tests = length(m_list);

            %% Preallocate properties
            obj.U_acc = zeros(num_tests, 2);
            obj.V_acc = zeros(num_tests, 2);
            obj.times = zeros(num_tests, 2);
            if strcmp(matr_type, 'custom sigmas')
                obj.s_acc = cell(num_tests, 2);
            end

            %% Start the tests
            for i = 1:num_tests
                m = m_list(i);
                n = n_list(i);

                if strcmp(matr_type, 'custom sigmas')
                    [obj, A] = obj.build_test_matrix( ...
                            matr_type, m, n, sigmas{i} ...
                    );
                else
                    [obj, A] = obj.build_test_matrix(matr_type, m, n);
                end

                I = eye(min(m,n));
                
                %% SVD - MATLAB
                tic;
                [U_matlab, s_matlab, V_matlab] = svd(A, 'econ', 'vector');
                obj.times(i,1) = toc;
                obj.U_acc(i,1) = norm(U_matlab' * U_matlab - I, 'inf');
                obj.V_acc(i,1) = norm(V_matlab' * V_matlab - I, 'inf');
                if strcmp(matr_type, 'custom sigmas')
                    obj.s_acc{i,1} = abs( (sigmas{i} - s_matlab) ./ sigmas{i} );
                end
                
                %% SVD - Custom
                tic;
                [U_custom, s_custom, V_custom] = svd_custom(A, tau);
                obj.times(i,2) = toc;
                obj.U_acc(i,2) = norm(U_custom' * U_custom - I, 'inf');
                obj.V_acc(i,2) = norm(V_custom' * V_custom - I, 'inf');
                if strcmp(matr_type, 'custom sigmas')
                    obj.s_acc{i,2} = abs( (sigmas{i} - s_custom) ./ sigmas{i} );
                end
            end
        end
    end


    methods (Access = private)
        function [obj, A] = build_test_matrix(obj, matr_type, m, n, sigmas)
            %BUILD_TEST_MATRIX Returns a matrix of the specified type/size.
            switch matr_type
                case 'hilb'
                    if m ~= n
                        warning( ...
                            "Hilbert matrices are square, but " + ...
                            "m ~= n; using min(m,n) for both dimensions"...
                        );
                    end
                    obj.type = 'Hilbert';
                    A = hilb(min(m,n));

                case 'pascal'
                    if m ~= n
                        warning( ...
                            "Pascal matrices are square, but " + ...
                            "m ~= n; using min(m,n) for both dimensions"...
                        );
                    end
                    obj.type = 'Pascal';
                    A = pascal(min(m,n));

                case 'orth'
                    if n > m
                        error( ...
                            "Can't build a wide matrix of " + ...
                            "orthonormal column vectors" ...
                        );
                    end
                    obj.type = 'orthogonal';
                    A = orth(rand(m,n));

                case 'custom sigmas'
                    if nargin < 4
                        error("Singular values' vector not provided");
                    elseif length(sigmas) ~= min(m,n)
                        error("Wrong amount of singular values to build A");
                    end
                    U = orth(rand(m));
                    V = orth(rand(n));
                    A = U * diag(sigmas) * V';

                    obj.type = 'custom sigmas';

                otherwise
                    error("The specified matrix type is not valid");
            end
        end
    end
end