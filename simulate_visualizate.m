% EIT Reconstruction using GN/cGN Solvers with Visualization
% Processes boundary voltage data and reconstructs conductivity using EIDORS

clc;
clear;

% User selects files to process
[filename, pathname] = uigetfile({'*.csv;*.xls;*.xlsx','Supported Files'}, 'Select files for processing', 'MultiSelect', 'on');
if isequal(filename, 0)
    disp('No files selected');
    return;
end

% Handle single or multiple file selection
if ischar(filename)
    fileall = {fullfile(pathname, filename)};
else
    fileall = strcat(pathname, filename);
end

% User selects save directory
save_path = uigetdir('', 'Select directory to save images and conductivity data');
if isequal(save_path, 0)
    disp('No save directory selected');
    return;
end

% Process each file
for num = 1:length(fileall)
    dir = fileall{num};

    % Read data based on file type
    [~,~,ext] = fileparts(dir);
    switch lower(ext)
        case '.csv'
            vi = readmatrix(dir);
            vi = vi(:, 1); % Keep only first column
        case {'.xls', '.xlsx'}
            [~, ~, raw] = xlsread(dir);
            vi = cell2mat(raw(:, 1)); % Keep only first column
        otherwise
            disp(['Unsupported file type: ', ext]);
            continue;
    end

    % User input for conductivity filename
    result_file_name = inputdlg('Enter conductivity filename (without extension):', 'Conductivity Filename', [1, 50]);
    if isempty(result_file_name)
        disp('User cancelled input');
        return;
    end

    % Select solver type
    solver_choice = questdlg('Select solver:', 'Solver Selection', 'GN','cGN','Cancel');
    if strcmp(solver_choice, 'Cancel')
        disp('Operation cancelled');
        return;
    end

    result_file_name = result_file_name{1};

    % Initialize conductivity range
    min_elec_data_all = Inf;
    max_elec_data_all = -Inf;
    elec_data_all = cell(size(fileall));

    % Model parameters
    cyl_shape = [0.08, 0.05, 0.003]; % Domain size [height, radius]
    nelec = 16; % Number of electrodes per ring
    ring_vert_pos = [0.0196, 0.0392, 0.0588]; % Electrode ring positions
    nrings = 3; % Number of electrode rings
    elec_shape = [0.001, 0, 0.002]; % Electrode shape

    % Create cylindrical model using netgen
    fmdl = ng_mk_cyl_models(cyl_shape, [nelec, ring_vert_pos], elec_shape);

    % Build common EIT model
    imdl = mk_common_model('b3cr', [16, 3]);
    imdl.fwd_model = fmdl;

    % Create EIDORS stimulation pattern
    imdl.fwd_model.stimulation = mk_stim_patterns(nelec, nrings, [0, 1], [0, 1], {'no_meas_current', 'rotate_meas'}, 0.03);

    % Configure solver based on user selection
    switch solver_choice
        case 'GN'
            % Gauss-Newton solver
            imdl_GN = imdl;
            imdl_GN.solve = @inv_solve_gn;
            imdl_GN.reconst_type = 'absolute';
            imdl_GN.jacobian_bkgnd.value = 0.1;
            imdl_GN.inv_solve_gn.max_iterations = 3;
            imdl_GN.RtR_prior = @prior_noser;
            hp = 1e-2;
            imdl_GN.hyperparameter.value = hp;
            img = inv_solve(imdl_GN, vi);
        case 'cGN'
            % Conjugate Gauss-Newton solver
            imdl_cGN = imdl;
            imdl_cGN.solve = @inv_solve_gn;
            imdl_cGN.reconst_type = 'absolute';
            imdl_cGN.jacobian_bkgnd.value = 0.1;
            imdl_cGN.inv_solve_gn.elem_working = 'log_conductivity';
            imdl_cGN.inv_solve_gn.max_iterations = 3;
            imdl_cGN.RtR_prior = @prior_noser;
            hp = 1e-2;
            imdl_cGN.hyperparameter.value = hp;
            img = inv_solve(imdl_cGN, vi);
    end

    % Save conductivity data
    elec_data_all{num} = img.elem_data;

    % Save reconstruction results
    result_file_path = fullfile(save_path, [result_file_name '.mat']);
    save(result_file_path, 'img');

    % Create EIDORS visualization
    figure;
    subplot(1, 2, 1)
    show_fem(img);
    eidors_colourbar(img);
    subplot(1, 2, 2)
    show_3d_slices(img, [0.0196, 0.0392, 0.0588]);

    % Save figure
    figName_eidors = fullfile(save_path, ['EIT_eidorsImage_' result_file_name '.fig']);
    saveas(gcf, figName_eidors, 'fig');
end