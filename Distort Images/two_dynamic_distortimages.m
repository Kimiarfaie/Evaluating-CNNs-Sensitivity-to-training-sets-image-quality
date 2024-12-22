% Setup
clear; clc;
gpuDevice(2);
addpath(genpath('code_imdistort'));

% Input folders for train and val
train_input_folder = 'path to ImageNet_subset/train';
val_input_folder = 'path to ImageNet_subset/val';

% Output folders for distorted train and val
train_output_folder = 'path to save Distorted_subset/train';
val_output_folder = 'path to save Distorted_subset/val';

% Distortion type and initial level
distortion_function = @imcolorsaturate; % Set the desired distortion function here (@imcolorsaturate or @imhueshift)

% Adjust initial level based on distortion function
if isequal(distortion_function, @hue_shift)
    initial_level = 0.01; % Hue shift initial level
    adjustment_step = 0.01; % Adjustment step for hue shift
elseif isequal(distortion_function, @imcolorsaturate)
    initial_level = 1.5; % Saturation adjustment initial level
    adjustment_step = 0.1; % Adjustment step for saturation adjustment
else
    error('Unsupported distortion function. Please use @hue_shift or @imcolorsaturate.');
end
max_adjustments = 136; % Adjustment limit
min_value = 6; % Minimum acceptable S-CIELAB value
max_value = 10; % Maximum acceptable S-CIELAB value

% Define Excel file paths
output_excel = 'DistortedImagesLog.xlsx';
skipped_excel = 'SkippedImagesLog.xlsx';

% Create headers and initialize an empty table for logging
headers = {'Image Name', 'Synset', 'Final Distortion Level', 'S-CIELAB'};
log_table = cell2table(cell(0, length(headers)), 'VariableNames', headers);
skipped_table = cell2table(cell(0, 2), 'VariableNames', {'Image Name', 'Reason'});

% Process train and val directories
[log_table, skipped_table] = process_directory(train_input_folder, train_output_folder, distortion_function, initial_level, adjustment_step, max_adjustments, log_table, skipped_table, min_value, max_value);
[log_table, skipped_table] = process_directory(val_input_folder, val_output_folder, distortion_function, initial_level, adjustment_step, max_adjustments, log_table, skipped_table, min_value, max_value);

% Save outputs
% Save log tables to Excel files
writetable(log_table, output_excel);
writetable(skipped_table, skipped_excel);
disp(['Metrics saved to Excel file: ', output_excel]);
disp(['Skipped images saved to Excel file: ', skipped_excel]);

%% Functions
% Process a directory of images (train/val)
% Process a directory of images (train/val)
function [log_table, skipped_table] = process_directory(input_folder, output_folder, distortion_function, initial_level, adjustment_step, max_adjustments, log_table, skipped_table, min_value, max_value)
    subfolders = dir(input_folder);
    subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name}, '.')); % Exclude '.' and '..'

    for i = 1:length(subfolders)
        synset_folder = subfolders(i).name;
        fprintf('\nProcessing synset: %s\n', synset_folder);

        % Input and output paths for the synset folder
        input_synset_folder = fullfile(input_folder, synset_folder);
        output_synset_folder = fullfile(output_folder, synset_folder);

        if ~exist(output_synset_folder, 'dir')
            mkdir(output_synset_folder);
            fprintf('Created output folder: %s\n', output_synset_folder);
        end

        % List all images in the current synset folder
        image_files = dir(fullfile(input_synset_folder, '*.JPEG'));

        if isempty(image_files)
            warning('No images found in folder: %s', input_synset_folder);
            continue;
        end

        % Loop through each image in the synset folder
        for img_idx = 1:length(image_files)
            img_name = image_files(img_idx).name;
            img_path = fullfile(input_synset_folder, img_name);
            fprintf('\nProcessing image: %s\n', img_path);

            % Read image
            try
                ref_im = im2double(imread(img_path));
            catch ME
                warning('Failed to read image: %s. Error: %s', img_name, ME.message);
                skipped_table = [skipped_table; {img_name, 'Read error'}];
                continue;
            end

            [~, basename, ~] = fileparts(img_name);

            % Apply initial distortion and compute S-CIELAB
            dist_level = initial_level;
            scielab_value = SCIELAB_FR(ref_im, distortion_function(ref_im, dist_level));

            % Adjust distortion until S-CIELAB is within range or max adjustments are reached
            adjustment_counter = 0;
            while (scielab_value < min_value || scielab_value > max_value) && adjustment_counter < max_adjustments
                dist_level = adjust_distortion_level(dist_level, scielab_value, min_value, max_value, adjustment_step);
                scielab_value = SCIELAB_FR(ref_im, distortion_function(ref_im, dist_level));
                adjustment_counter = adjustment_counter + 1;
            end

            % Save the distorted image if within range
            if scielab_value >= min_value && scielab_value <= max_value
                output_path = fullfile(output_synset_folder, sprintf('%s.png', basename));
                imwrite(distortion_function(ref_im, dist_level), output_path);
                log_table = [log_table; {img_name, synset_folder, dist_level, scielab_value}];
            else
                skipped_table = [skipped_table; {img_name, 'Exceeded max adjustments'}];
            end
        end
    end
end

% Helper function to adjust the distortion level based on S-CIELAB
function new_level = adjust_distortion_level(current_level, scielab_value, min_value, max_value, adjustment_step)
    if scielab_value < min_value 
        new_level = current_level + adjustment_step; % Increase level
    elseif scielab_value > max_value 
        new_level = current_level - adjustment_step; % Decrease level
    else
        new_level = current_level; % Keep current level if within range
    end
end
