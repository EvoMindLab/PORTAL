%********************************** PORTAL Save Instance Function **********************************
%Author: Generated for PORTAL Framework
%Last Edited: 2025-10-10
%Title: Save PORTAL Instance to JSON
% --------
% Description:
%   Saves a PORTAL benchmark instance to a JSON file in the Instances directory
% --------
% Inputs:
%   PORTAL - PORTAL benchmark instance structure
%   filename - (optional) Filename for the JSON file (default: auto-generated)
% Output:
%   filepath - Full path to the saved file
%************************************************************************************************

function filepath = saveInstance(PORTAL, filename)
    %% Determine filename
    if nargin < 2 || isempty(filename)
        filename = sprintf('PORTAL_d%d_K%d_seed%d.json', PORTAL.d, PORTAL.K, PORTAL.Seed);
    end

    % Ensure .json extension
    [~, name, ext] = fileparts(filename);
    if ~strcmp(ext, '.json')
        filename = [name '.json'];
    end

    %% Determine save path
    script_dir = fileparts(mfilename('fullpath'));
    instances_dir = fullfile(script_dir, 'Instances');

    % Create Instances directory if it doesn't exist
    if ~exist(instances_dir, 'dir')
        mkdir(instances_dir);
    end

    filepath = fullfile(instances_dir, filename);

    %% Convert PORTAL structure to JSON-compatible format
    % MATLAB's jsonencode doesn't handle RandStream objects
    PORTAL_save = PORTAL;

    % Remove RandStream object
    if isfield(PORTAL_save, 'Rng')
        PORTAL_save = rmfield(PORTAL_save, 'Rng');
    end

    % Convert Psi_deg (rotation angles) to cell array for JSON
    if isfield(PORTAL_save, 'P') && isfield(PORTAL_save.P, 'Psi_deg')
        Psi_deg_temp = PORTAL_save.P.Psi_deg;
        Psi_deg_cell = cell(1, PORTAL.K);
        for k = 1:PORTAL.K
            Psi_deg_cell{k} = Psi_deg_temp(:,:,k);
        end
        PORTAL_save.P.Psi_deg_cell = Psi_deg_cell;
        PORTAL_save.P = rmfield(PORTAL_save.P, 'Psi_deg');
    end

    % Remove R matrices (rotation matrices) - will be computed from Psi_deg when loaded
    if isfield(PORTAL_save, 'P') && isfield(PORTAL_save.P, 'R')
        PORTAL_save.P = rmfield(PORTAL_save.P, 'R');
    end

    %% Save to JSON
    try
        json_str = jsonencode(PORTAL_save, 'PrettyPrint', true);
        fid = fopen(filepath, 'w');
        if fid == -1
            error('Cannot open file for writing: %s', filepath);
        end
        fprintf(fid, '%s', json_str);
        fclose(fid);
        fprintf('Instance saved successfully to:\n  %s\n', filepath);
    catch ME
        error('Failed to save instance: %s', ME.message);
    end
end