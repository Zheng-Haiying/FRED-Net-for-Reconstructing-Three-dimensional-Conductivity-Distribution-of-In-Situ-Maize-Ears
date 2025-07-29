% EIT Dataset Generation for Maize Training Data
% Generates conductivity distributions and boundary voltage data for 3D cylindrical EIT model
% Absolute imaging with different noise levels (None, 20dB, 30dB, 40dB, 50dB SNR)
% Background conductivity: 0.035 S/m, Object conductivity: 0.00000001 S/m
% Excitation current: 0.025 A

clc
clear
close

num = 10000; % Number of samples
r = 0.0355; % XY plane radius

% Create output directory
base_dir = 'dataset save path';
if ~exist(base_dir, 'dir')
    mkdir(base_dir);
end

% Model parameters
cyl_shape = [0.066, r, 0.003]; % Domain size [height, radius]
nelec = 16; % Number of electrodes per ring
ring_vert_pos = [0.011, 0.033, 0.055]; % Electrode ring positions
nrings = 3; % Number of electrode rings
elec_shape = [0.001, 0, 0.02]; % Electrode shape

% Create cylindrical model using netgen
fmdl = ng_mk_cyl_models(cyl_shape, [nelec, ring_vert_pos], elec_shape);

% Build common EIT model
imdl = mk_common_model('b3cr', [nelec, nrings]);
imdl.fwd_model = fmdl;

% Create EIDORS stimulation pattern
imdl.fwd_model.stimulation = mk_stim_patterns(nelec, nrings, [0, 1], [0, 1], {'no_meas_current', 'rotate_meas'}, 0.03);

img1 = mk_image(imdl, 1); % Background conductivity

% Extract electrode coordinates
num_elec = length(img1.fwd_model.electrode);
elec_coords = zeros(num_elec, 3);
for i = 1:num_elec
    node_idx = img1.fwd_model.electrode(i).nodes;
    elec_coords(i, :) = mean(img1.fwd_model.nodes(node_idx, :), 1);
end
electrode_positions = elec_coords;

% Initialize sample arrays
DDL_samples = zeros(length(img1.elem_data), num);
BV_samples = zeros(2160, num);
BV_20dB_samples = zeros(2160, num);
BV_30dB_samples = zeros(2160, num);
BV_40dB_samples = zeros(2160, num);
BV_50dB_samples = zeros(2160, num);

% Process samples in chunks for Vol_samples
chunk_size = 1;
num_chunks = ceil(num / chunk_size);

for chunk = 1:num_chunks
    start_idx = (chunk - 1) * chunk_size + 1;
    end_idx = min(chunk * chunk_size, num);

    chunk_data = zeros(length(img1.elem_data), 48, chunk_size);

    for k = start_idx:end_idx
        disp(['Processing sample: ' num2str(k)]);

        % Conductivity distribution parameters
        base_scale_x = [5, 10];
        base_scale_y = [5, 10];
        base_scale_z = [4, 9];
        diffusion_range = [1, 10];
        center_position_perturbation = [0.01, 0.3];
        distance_perturbation_scale = [0.001, 0.03];
        min_conductivity_range = [1, 2.5];
        max_conductivity_range = [3.5, 4.5];
        num_centers_z_range = [10, 40];
        num_centers_xy_range = 30;
        num_directions_range = [8, 32];
        conductivity_fluctuation_factor = 0.5;

        % Calculate element centroids
        elem_centroids = (fmdl.nodes(fmdl.elems(:,1), :) + fmdl.nodes(fmdl.elems(:,2), :) + ...
                         fmdl.nodes(fmdl.elems(:,3), :) + fmdl.nodes(fmdl.elems(:,4), :)) / 4;

        % Find domain boundaries
        min_coords = min(elem_centroids);
        max_coords = max(elem_centroids);
        range_coords = max_coords - min_coords;

        % Generate Z-direction centers
        num_centers_z = randi(num_centers_z_range);
        center_z = min_coords(3) + (0.4 + (center_position_perturbation(2)-center_position_perturbation(1))*rand(1, num_centers_z) + center_position_perturbation(1)) * range_coords(3);

        % Initialize conductivity
        elems = zeros(size(elem_centroids, 1), 1);

        % Generate multiple directions in XY plane
        num_directions = randi(num_directions_range);
        directions = linspace(0, 2*pi, num_directions+1);

        % Generate diffusion speeds for different directions
        scale_x_min = base_scale_x(1) + rand(1, num_directions) * (base_scale_x(2) - base_scale_x(1));
        scale_x_max = base_scale_x(1) + rand(1, num_directions) * (base_scale_x(2) - base_scale_x(1));
        scale_y_min = base_scale_y(1) + rand(1, num_directions) * (base_scale_y(2) - base_scale_y(1));
        scale_y_max = base_scale_y(1) + rand(1, num_directions) * (base_scale_y(2) - base_scale_y(1));

        % Process each Z center
        for i = 1:length(center_z)
            num_centers_xy = randi(num_centers_xy_range);

            for j = 1:num_centers_xy
                center_xy = [min_coords(1) + (0.4 + (center_position_perturbation(2)-center_position_perturbation(1))*rand() + center_position_perturbation(1)) * range_coords(1),
                             min_coords(2) + (0.4 + (center_position_perturbation(2)-center_position_perturbation(1))*rand() + center_position_perturbation(1)) * range_coords(2)];
                center_xy = center_xy';

                % Calculate distances with perturbation
                distances_xy = sqrt(sum((elem_centroids(:,1:2) - center_xy).^2, 2)) + (distance_perturbation_scale(2)-distance_perturbation_scale(1))*randn(size(elem_centroids, 1), 1) + distance_perturbation_scale(1);
                distance_z = abs(elem_centroids(:,3) - center_z(i));

                % Calculate conductivity effect
                elem_effect = exp(-distances_xy .* scale_x_min) .* exp(-distances_xy .* scale_y_min) .* exp(-distance_z / base_scale_z(1));

                % Apply directional decay factors
                for q = 1:num_directions
                    direction = directions(q);

                    min_decay_factor = (min_conductivity_range(1) + rand() * (min_conductivity_range(2) - min_conductivity_range(1))) * (1 - conductivity_fluctuation_factor);
                    max_decay_factor = (max_conductivity_range(1) + rand() * (max_conductivity_range(2) - max_conductivity_range(1))) * (1 + conductivity_fluctuation_factor);

                    decay_factor = min_decay_factor + (max_decay_factor - min_decay_factor) * sin(direction);

                    max_distance = max(distances_xy);
                    decay_factor = decay_factor * (1 - (distances_xy - (4/6) * max_distance) / (1/6) * max_distance);
                    decay_factor(decay_factor < 0) = 0;
                    elems = elems + elem_effect .* decay_factor;
                end
            end
        end

        % Normalize conductivity range
        normalized_elems = (elems - min(elems)) / (max(elems) - min(elems));
        min_conductivity = min_conductivity_range(1) + rand() * (min_conductivity_range(2) - min_conductivity_range(1));
        max_conductivity = max_conductivity_range(1) + rand() * (max_conductivity_range(2) - max_conductivity_range(1));
        elems = min_conductivity + normalized_elems * (max_conductivity - min_conductivity);
        img1.elem_data = elems;

        % Forward solve
        img1.fwd_solve.get_all_nodes = 1;
        vi = fwd_solve(img1);

        % Extract node voltages and element information
        node_voltages = vi.volt;
        elems_index = img1.fwd_model.elems;

        % Calculate element voltages
        num_elems = size(elems_index, 1);
        num_frames = size(node_voltages, 2);
        elem_voltages = zeros(num_elems, num_frames);

        for s = 1:num_elems
            node_indices = elems_index(s, :);
            elem_voltages(s, :) = mean(node_voltages(node_indices, :), 1);
        end

        chunk_data(:, :, k - start_idx + 1) = elem_voltages;

        % Save conductivity and boundary voltage data
        DDL_samples(:, k) = img1.elem_data(:, 1);
        BV_samples(:, k) = vi.meas(:, 1);

        % Add noise at different SNR levels
        vi_20dB = add_noise(10^(20/20), vi);
        BV_20dB_samples(:, k) = vi_20dB.meas(:, 1);

        vi_30dB = add_noise(10^(30/20), vi);
        BV_30dB_samples(:, k) = vi_30dB.meas(:, 1);

        vi_40dB = add_noise(10^(40/20), vi);
        BV_40dB_samples(:, k) = vi_40dB.meas(:, 1);

        vi_50dB = add_noise(10^(50/20), vi);
        BV_50dB_samples(:, k) = vi_50dB.meas(:, 1);
    end

    % Save chunk data
    chunk_filename = [base_dir, '/', num2str(start_idx - 1), '_', num2str(end_idx - 1), '_Vol_samples.h5'];
    h5create(chunk_filename, '/Vol_samples', size(chunk_data), 'Datatype', 'double');
    h5write(chunk_filename, '/Vol_samples', chunk_data);
end

% Save DDL samples
DDL_filename = [base_dir, '/DDL_samples.h5'];
h5create(DDL_filename, '/DDL_samples', size(DDL_samples), 'Datatype', 'double');
h5write(DDL_filename, '/DDL_samples', DDL_samples);

% Save other datasets
other_datasets = {'BV_samples', 'BV_20dB_samples', 'BV_30dB_samples', 'BV_40dB_samples', 'BV_50dB_samples'};
for i = 1:numel(other_datasets)
    data_name = other_datasets{i};
    data = eval([data_name, ';']);
    data_filename = [base_dir, '/', data_name, '.h5'];
    h5create(data_filename, ['/', data_name], size(data), 'Datatype', 'double');
    h5write(data_filename, ['/', data_name], data);
end

% Export coordinates and electrode positions
Coordinates_3D_dir = [base_dir '/'];
if ~exist(Coordinates_3D_dir, 'dir')
    mkdir(Coordinates_3D_dir);
end

% Export element centroids
nodes = img1.fwd_model.nodes;
elems = img1.fwd_model.elems;
elem_centroids = (nodes(elems(:,1), :) + nodes(elems(:,2), :) + nodes(elems(:,3), :) + nodes(elems(:,4), :)) / 4;
writematrix(elem_centroids, strcat(Coordinates_3D_dir, 'Coordinates_3D', '.csv'));

% Export electrode positions
electrode_positions_filename = [Coordinates_3D_dir, 'Electrode_positions.csv'];
writematrix(electrode_positions, electrode_positions_filename);