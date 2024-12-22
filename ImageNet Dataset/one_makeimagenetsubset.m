
clear all; clc;
gpuDevice(2);

% Define paths
mapping_file = 'path to LOC_synset_mapping.txt'; 
val_solution_file = 'path to LOC_val_solution.csv';
source_train_folder = 'path to ILSVRC/Data/CLS-LOC/train';    
source_val_folder = 'path to ILSVRC/Data/CLS-LOC/val';         
destination_folder = 'path to destination folder';
num_train = 100;                                  
num_val = 20;       

% Create a new mapping of the 100 classes 
new_mapping_file = fullfile(destination_folder, 'LOC_synset_mapping.txt'); % New mapping file

% Define destination folders for train and val
destination_train_folder = fullfile(destination_folder, 'train');
destination_val_folder = fullfile(destination_folder, 'val');

% Read the LOC_synset_mapping.txt file
fid = fopen(mapping_file, 'r');
synset_mapping = textscan(fid, '%s %[^\n]', 'Delimiter', ' '); % Read synset and class names, separated by a space
fclose(fid);

% Extract synset IDs and class names
synsets = synset_mapping{1}; % Synsets are in the first column
class_names = synset_mapping{2}; % Class names are in the second column

% Clean any trailing whitespace from synset IDs (removes any accidental class name sticking)
synsets = strtrim(synsets); 

% Get the last 100 synsets and their corresponding class names
last_100_synsets = synsets(end-99:end); % Correctly extract the last 100 synsets
last_100_class_names = class_names(end-99:end);

% Write the new mapping file with synset IDs and class names
fid_new = fopen(new_mapping_file, 'w');
for i = 1:length(last_100_synsets)
    fprintf(fid_new, '%s, %s\n', last_100_synsets{i}, last_100_class_names{i}); % Write synset and corresponding class names
end
fclose(fid_new);

disp('New mapping file created successfully.');

%%
% Create the destination directories for train and val data if they don't exist

if ~exist(destination_train_folder, 'dir')
    mkdir(destination_train_folder);
end

if ~exist(destination_val_folder, 'dir')
    mkdir(destination_val_folder);
end

% Loop through each of the last 100 synsets
for i = 1:length(last_100_synsets)
    synset = last_100_synsets{i}; % Ensure only the synset ID is used
    fprintf('Processing synset: %s\n', synset);
    
    % Create the destination directories for this synset in both train and val (use only synset ID)
    train_synset_dest = fullfile(destination_train_folder, synset); % Use only synset ID for folder names
    val_synset_dest = fullfile(destination_val_folder, synset);     % Use only synset ID for folder names
    
    if ~exist(train_synset_dest, 'dir')
        mkdir(train_synset_dest);
    end
    
    if ~exist(val_synset_dest, 'dir')
        mkdir(val_synset_dest);
    end
    
    % Train images (Look for the synset folder in the source)
    source_train_class_folder = fullfile(source_train_folder, synset); % Use only synset ID
    train_images = dir(fullfile(source_train_class_folder, '*.JPEG')); % Get all train images
    
    colorful_count = 0; % Counter for colorful images
    
    % Copy the first 'num_train' colorful images to the destination folder
    for j = 1:length(train_images)
        if colorful_count >= num_train
            break; % Stop once we have enough colorful images
        end
        
        src_image = fullfile(source_train_class_folder, train_images(j).name);
        image = imread(src_image);
        
        % Check if the image is colorful - you can skip this
        if size(image, 3) == 3 && check_if_colorful(image)
            colorful_count = colorful_count + 1;
            dest_image = fullfile(train_synset_dest, train_images(j).name); % Keep original file names
            copyfile(src_image, dest_image);
        end
    end
    
    % Process validation images for the current synset
    val_solution = readtable(val_solution_file, 'Delimiter', ',');
    val_image_ids = val_solution.ImageId; % Image IDs from the val_solution file
    val_predictions = val_solution.PredictionString; % Corresponding synset labels
    
    % Find the indices of images belonging to this synset
    class_val_indices = find(contains(val_predictions, synset)); % Get indices of val images for this synset
    
    colorful_val_count = 0;
    
    % Copy the first 'num_val' colorful validation images to the destination folder
    for j = 1:length(class_val_indices)
        if colorful_val_count >= num_val
            break; % Stop once we have enough colorful val images
        end
        
        val_image_name = val_image_ids{class_val_indices(j)}; % Get the validation image ID
        src_val_image = fullfile(source_val_folder, [val_image_name '.JPEG']); % Validation image file path
        
        % Check if the image is colorful
        val_image = imread(src_val_image);
        if size(val_image, 3) == 3 && check_if_colorful(val_image)
            colorful_val_count = colorful_val_count + 1;
            dest_val_image = fullfile(val_synset_dest, [val_image_name '.JPEG']); % Keep original file names
            copyfile(src_val_image, dest_val_image);
        end
    end
end

disp('Processing complete. Colorful validation and train images have been organized using synset folders.');

%% Creating a new mapping file for the new dataset
% Define paths
original_val_solution = 'path to Full ImageNet LOC_val_solution.csv';
destination_val_folder = 'path to new subset val'; % Where the subset val images are stored
new_val_solution_file = 'path to save LOC_val_solution.csv'; % New file

% Read the original LOC_val_solution.csv
val_solution = readtable(original_val_solution, 'Delimiter', ',');
 
% Initialize a new table for the subset
new_val_solution = table();

% Get all synset folders in the validation subset directory
synset_folders = dir(destination_val_folder);
synset_folders = synset_folders([synset_folders.isdir] & ~startsWith({synset_folders.name}, '.')); % Exclude '.' and '..'

% Loop through each synset folder and collect the image IDs
for i = 1:length(synset_folders)
    synset = synset_folders(i).name; % Synset folder name (e.g., 'n12345678')
    fprintf('Processing synset: %s\n', synset);

    % Get all image files in this synset folder
    image_files = dir(fullfile(destination_val_folder, synset, '*.JPEG'));

    % Extract the image IDs (e.g., 'ILSVRC2012_val_00000001')
    image_ids = cellfun(@(x) strtok(x, '.'), {image_files.name}, 'UniformOutput', false);

    % Find the corresponding entries in the original val solution file
    for j = 1:length(image_ids)
        img_id = image_ids{j};

        % Find the row in the original val solution corresponding to this image ID
        row_idx = find(strcmp(val_solution.ImageId, img_id));
        if ~isempty(row_idx)
            % Add this row to the new table
            new_val_solution = [new_val_solution; val_solution(row_idx, :)];
        else
            warning('Image ID %s not found in the original val solution.', img_id);
        end
    end
end

% Write the new val solution table to a CSV file
writetable(new_val_solution, new_val_solution_file);

disp('New LOC_val_solution_subset.csv created successfully.');
%%

% Function to check if an image is colorful; only for our purpose we want images with three channels, you may skip this based on your needs

function is_colorful = check_if_colorful(image)
    % Compare the R, G, and B channels
    red_channel = image(:, :, 1);
    green_channel = image(:, :, 2);
    blue_channel = image(:, :, 3);
    
    % If all three channels are identical, the image is grayscale
    is_colorful = ~(isequal(red_channel, green_channel) && isequal(green_channel, blue_channel));
end