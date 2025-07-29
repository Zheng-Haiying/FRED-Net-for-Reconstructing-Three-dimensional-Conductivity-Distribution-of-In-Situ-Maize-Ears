% Deep Learning Model Conductivity Visualization for Acrylic Objects
% Visualizes predicted conductivity distributions from deep learning models

clc;
clear;

% User selects conductivity files to process
[filename, pathname] = uigetfile({'*.csv;*.xls;*.xlsx','Supported Files'}, 'Select conductivity files for processing', 'MultiSelect', 'on');
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

% User selects save directory for images
save_path = uigetdir('', 'Select directory to save images');
if isequal(save_path, 0)
    disp('No save directory selected');
    return;
end

% Process each file
for num = 1:length(fileall)
    dir = fileall{num};

    % Read conductivity data based on file type
    [~,~,ext] = fileparts(dir);
    switch lower(ext)
        case '.csv'
            ddl = readmatrix(dir);
            ddl = ddl'; % Transpose data
        case {'.xls', '.xlsx'}
            [~, ~, raw] = xlsread(dir);
            ddl = cell2mat(raw');
        otherwise
            disp(['Unsupported file type: ', ext]);
            continue;
    end

    % Initialize conductivity range variables
    min_elec_data_all = Inf;
    max_elec_data_all = -Inf;
    elec_data_all = ddl;

    % Model parameters
    nelec = 16; % Number of electrodes per ring
    ring_vert_pos = [0.0196, 0.0392, 0.0588]; % Electrode ring positions
    nrings = 3; % Number of electrode rings
    elec_shape = [0.001, 0, 0.02]; % Electrode shape

    % Create forward model
    fmdl = ng_mk_cyl_models([0.08, 0.05, 0.004], [nelec, ring_vert_pos], elec_shape);
    imdl = mk_common_model('b3cr', [16, 3]);
    imdl.fwd_model = fmdl;
    imdl.fwd_model.stimulation = mk_stim_patterns(nelec, nrings, [0, 1], [0, 1], {'no_meas_current', 'rotate_meas'}, 0.03);

    % Create image with conductivity data
    img = mk_image(imdl, elec_data_all);

    % Generate visualizations
    [~, name, ~] = fileparts(dir); % Extract filename for image naming

    figure;
    % 3D FEM visualization
    subplot(1, 3, 1)
    show_fem(img);
    eidors_colourbar(img);

    % 3D slice visualization
    subplot(1, 3, 2)
    show_3d_slices(img, ring_vert_pos);
    eidors_colourbar(img);

    % 2D slice visualization at specific levels
    subplot(1, 3, 3)
    levels = [inf, inf, 0.013;
              inf, inf, 0.026;
              inf, inf, 0.039;
              inf, inf, 0.052;
              inf, inf, 0.065;
              inf, inf, 0.078];

    % Set resolution for color mapping
    img.calc_colours.npoints = 256;
    show_slices(img, levels);
    eidors_colourbar(img);

    % Save figure
    figName_eidors = fullfile(save_path, ['EIT_3DImage_' name '.fig']);
    saveas(gcf, figName_eidors, 'fig');
end