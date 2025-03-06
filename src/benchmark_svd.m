clear;
close all;

%% Workspace variables
tau = 1e-8; % See 'help eig_tridiag' for details


%% Tests for orthogonal matrices
n_intervalL = 50;
n_step = 50;
n_intervalR = 800;
n_list = n_intervalL: n_step: n_intervalR;

% Initialize the objects
orth_results = BenchmarkSVD('orth', n_list, n_list, tau);

% Plot results
plot_results(orth_results, n_list);


%% Tests for Hilbert and Pascal matrices
n_intervalL = 2;
n_step = 1;
n_intervalR = 100;
n_list = n_intervalL: n_step: n_intervalR;

% Initialize the objects
hilb_results = BenchmarkSVD('hilb', n_list, n_list, tau);
pasc_results = BenchmarkSVD('pascal', n_list, n_list, tau);

% Plot results
plot_results(hilb_results, n_list);
plot_results(pasc_results, n_list);


%% Tests for custom sigmas matrices
n_size = 16;

n_list = [n_size, n_size, n_size, n_size];
sigmas = {
    logspace(n_size/8, 1, n_size)';     % cond(A) = 1e02
    logspace(n_size/4, 1, n_size)';     % cond(A) = 1e04
    logspace(n_size/2, 1, n_size)';     % cond(A) = 1e08
    logspace(n_size, 1, n_size)';       % cond(A) = 1e16
};
cond_values = {'10^2', '10^4', '10^8', '10^{16}'};

% Initialize the object
sigmas_results = BenchmarkSVD('custom sigmas', n_list, n_list, tau, sigmas);

% Plot results
plot_results_sigmas(sigmas_results, sigmas, cond_values);


%% Plot funcs
function plot_results(svd_results, n_list)
plot_type_names =  {'U precision', 'V precision', 'Performance'};
title_template = "%s results for %s matrices";
label_x = "Matrix size (n)";
labels_y = { ...
    "$|\!|U^{T}U - I|\!|_{\infty}$", ...
    "$|\!|V^{T}V - I|\!|_{\infty}$", ...
    "Time (s)"
};
legend_names = {'svd()', 'svd\_custom()'};

prop_names = properties(svd_results);
prop_names = prop_names(3:end); % Exclude the 'type' and 's_acc' properties

for i = 1:length(plot_type_names)
    plot_title = sprintf( ...
        title_template, ...
        plot_type_names{i}, ...
        svd_results.type ...
    );

    figure('name', plot_title);
    semilogy(n_list, svd_results.(prop_names{i}));
    grid on;
    title(plot_title);
    xlabel(label_x, 'FontSize', 14);
    ylabel(labels_y{i}, 'Interpreter','latex', 'FontSize', 18);

    legend(legend_names, 'Location', 'northwest');

    save_curr_plot_img(plot_title);
end

end


function plot_results_sigmas(svd_results, sigmas, cond_values)
% If the test matrix has been built with known singular values, plot the
% singular values' relative errors
if ~strcmp(svd_results.type, 'custom sigmas')
    error( ...
        "This function only plots relative errors for matrices " + ...
        "of type 'custom sigmas'" ...
        );
end

for i = 1:length(cond_values)
    label_x = "${\sigma}_{i}$";
    label_y = "$|\frac{{\sigma}_{i} - \tilde{{\sigma}}_{i}}{{\sigma}_{i}}|$";

    plot_title = "Relative errors for singular values - k_2(A) = " + cond_values{i};
    legend_names = {'svd()', 'svd\_custom()'};

    figure('name', plot_title);
    loglog(sigmas{i}, [svd_results.s_acc{i,1}, svd_results.s_acc{i,2}]);
    grid on;
    title(plot_title);
    xlabel(label_x, 'Interpreter','latex', 'FontSize', 18);
    ylabel(label_y, 'Interpreter','latex', 'FontSize', 18);

    legend(legend_names, 'Location', 'north');

    save_curr_plot_img(plot_title);
end

end


function save_curr_plot_img(plot_title)
% Set this to true to save plot images in .eps format
save_plot_imgs = false;
plot_imgs_path = './plot_imgs';

if save_plot_imgs && ~exist(plot_imgs_path, 'dir')
    mkdir(plot_imgs_path);
end

if save_plot_imgs
    curr_plt = gcf;
    removeToolbarExplorationButtons(curr_plt);
    img_name = sprintf( ...
        "%s/%.2d - %s.eps", ...
        plot_imgs_path, ...
        curr_plt.Number, ...
        plot_title ...
        );
    img_name = strrep(img_name, ' ', '_');
    exportgraphics(curr_plt, img_name, 'ContentType','vector');
end

end