%********************************** PORTAL Fitness Function **********************************
% Author: Mai Peng, Danial Yazdani, Delaram Yazdani
% Email: pengmai1998@gmail.com
%        danial.yazdani@gmail.com
%        delaram.yazdani@yahoo.com
% Last Edited: 2025-11-28
%Title: Load PORTAL Instance from JSON
% --------
% Description:
%   Loads a PORTAL benchmark instance from a JSON file
% --------
% Inputs:
%   filename - Filename or full path to the JSON file
% Output:
%   PORTAL - PORTAL benchmark instance structure
%************************************************************************************************

function PORTAL = load_instance(filename)
    %% Determine file path
    if ~exist(filename, 'file')
        % Try in Instances directory
        script_dir = fileparts(mfilename('fullpath'));
        parent_dir = fileparts(script_dir); 
        instances_dir = fullfile(parent_dir, 'Instances');
        filepath = fullfile(instances_dir, filename);

        if ~exist(filepath, 'file')
            error('Instance file not found: %s', filename);
        end
    else
        filepath = filename;
    end

    %% Load from JSON
    try
        fid = fopen(filepath, 'r');
        if fid == -1
            error('Cannot open file for reading: %s', filepath);
        end
        json_str = fread(fid, '*char')';
        fclose(fid);

        PORTAL = jsondecode(json_str);
        fprintf('Instance loaded successfully from:\n  %s\n', filepath);
    catch ME
        error('Failed to load instance: %s', ME.message);
    end

    %% Reconstruct RandStream
    if isfield(PORTAL, 'Seed')
        PORTAL.Rng = RandStream('mt19937ar', 'Seed', PORTAL.Seed);
    else
        warning('Seed not found in loaded instance. Creating new RandStream.');
        PORTAL.Rng = RandStream('mt19937ar', 'Seed', 1234);
    end

    %% Reconstruct rotation angles from Psi_deg_cell
    if isfield(PORTAL, 'P') && isfield(PORTAL.P, 'Psi_deg_cell')
        Psi_deg_cell = PORTAL.P.Psi_deg_cell;
        K = PORTAL.K;
        d = PORTAL.d;
        Psi_deg_temp = zeros(d, d, K);

        if iscell(Psi_deg_cell)
            % If it's a cell array
            for k = 1:K
                matrix_k = Psi_deg_cell{k};
                if iscell(matrix_k)
                    % Nested cell (from JSON): convert rows to matrix
                    for row = 1:d
                        Psi_deg_temp(row, :, k) = matrix_k{row};
                    end
                elseif isnumeric(matrix_k)
                    % Already a matrix
                    Psi_deg_temp(:,:,k) = matrix_k;
                end
            end
        elseif isnumeric(Psi_deg_cell)
            % jsondecode converted to numeric array
            Psi_size = size(Psi_deg_cell);
            Psi_ndims = ndims(Psi_deg_cell);

            % Case 2: 3D array [K×d×d] (actual jsondecode behavior) - CHECK FIRST!
            if Psi_ndims == 3 && Psi_size(1) == K && Psi_size(2) == d && Psi_size(3) == d
                % Extract each component and transpose dimensions
                for k = 1:K
                    Psi_deg_temp(:,:,k) = squeeze(Psi_deg_cell(k,:,:));
                end
            % Case 1: 3D array [d×d×K] (expected from save_instance) - only if K≠d
            elseif Psi_ndims == 3 && Psi_size(1) == d && Psi_size(2) == d && Psi_size(3) == K && K ~= d
                Psi_deg_temp = Psi_deg_cell;
            % Case 3: 2D array when K=1 [d×d]
            elseif Psi_ndims == 2 && K == 1 && Psi_size(1) == d && Psi_size(2) == d
                Psi_deg_temp(:,:,1) = Psi_deg_cell;
            % Case 4: Alternative dimension ordering
            elseif numel(Psi_deg_cell) == d * d * K
                % Try to reshape - assume it's the right data, wrong shape
                Psi_deg_temp = reshape(Psi_deg_cell, d, d, K);
            else
                error('Cannot interpret Psi_deg_cell: size=[%s], expected [%d×%d×%d] or [%d×%d×%d]', ...
                      num2str(Psi_size), d, d, K, K, d, d);
            end
        else
            error('Unexpected Psi_deg_cell type: %s', class(Psi_deg_cell));
        end

        PORTAL.P.Psi_deg = Psi_deg_temp;
        PORTAL.P = rmfield(PORTAL.P, 'Psi_deg_cell');

        % Compute rotation matrices R from Psi_deg
        R_temp = zeros(d, d, K);
        for k = 1:K
            Psi_rad = deg2rad(Psi_deg_temp(:,:,k));
            R_temp(:,:,k) = givens_from_psi(Psi_rad);
        end
        PORTAL.P.R = R_temp;
    end

    %% Fix all parameter dimensions (especially for K=1 case)
    PORTAL.P = fixParameterDimensions(PORTAL.P, PORTAL.K, PORTAL.d);

    %% Display instance info
    fprintf('Loaded PORTAL Instance:\n');
    fprintf('  Dimension: %d\n', PORTAL.d);
    fprintf('  Components: %d\n', PORTAL.K);
    fprintf('  Seed: %d\n', PORTAL.Seed);

    if isfield(PORTAL, 'baseline')
        fprintf('  Baselines: ');
        for k = 1:PORTAL.K
            if iscell(PORTAL.baseline)
                fprintf('%s ', PORTAL.baseline{k});
            else
                fprintf('%s ', PORTAL.baseline(k));
            end
        end
        fprintf('\n');
    end

    % Display rotation mode if available
    if isfield(PORTAL, 'transform_seq')
        fprintf('  Transformations: ');
        for k = 1:PORTAL.K
            fprintf("{");
            if iscell(PORTAL.transform_seq)
                for m = 1:length(PORTAL.transform_seq{k})
                    fprintf("'%s'", char(PORTAL.transform_seq{k}{m}));
                    if m ~= length(PORTAL.transform_seq{k})
                        fprintf("→");
                    end
                end
            end
            fprintf("}");
        end
        fprintf('\n');
    end
end

%% Helper function to fix parameter dimensions after JSON load
function P = fixParameterDimensions(P, K, d)
    % When K=1, jsondecode may compress dimensions. This function ensures
    % all parameters have the correct shape.

    % Fix K×d matrices (should be K rows, d columns)
    kd_fields = {'c', 'kappa_plus', 'kappa_minus', 'p_plus', 'p_minus', 'rho_plus', 'rho_minus'};
    for i = 1:length(kd_fields)
        field = kd_fields{i};
        if isfield(P, field) && ~isempty(P.(field)) && numel(P.(field)) > 0
            if ~isnan(P.(field)(1))
                P.(field) = ensureSize(P.(field), K, d);
            end
        end
    end

    % Fix K×1 vectors (should be K rows, 1 column)
    k1_fields = {'beta', 'p_single', 'Delta', 'r_ref', 'rho_single', 'form'};
    for i = 1:length(k1_fields)
        field = k1_fields{i};
        if isfield(P, field) && ~isempty(P.(field)) && numel(P.(field)) > 0
            if ~isnan(P.(field)(1))
                P.(field) = ensureSize(P.(field), K, 1);
            end
        end
    end

    % Fix 3D arrays: Psi_deg and R (d×d×K)
    if isfield(P, 'Psi_deg')
        P.Psi_deg = ensureSize3D(P.Psi_deg, d, d, K);
    end

    if isfield(P, 'R')
        P.R = ensureSize3D(P.R, d, d, K);
    end

    % Fix transform parameters
    if isfield(P, 'T_additive_periodic')
        T = P.T_additive_periodic;
        T.mu_plus = ensureSize(T.mu_plus, K, d);
        T.mu_minus = ensureSize(T.mu_minus, K, d);
        T.gamma_plus = ensureSize(T.gamma_plus, K, d);
        T.gamma_minus = ensureSize(T.gamma_minus, K, d);
        T.omega_plus = ensureSize(T.omega_plus, K, d);
        T.omega_minus = ensureSize(T.omega_minus, K, d);
        P.T_additive_periodic = T;
    end

    if isfield(P, 'T_log_sinusoidal')
        T = P.T_log_sinusoidal;
        T.mu_plus = ensureSize(T.mu_plus, K, d);
        T.mu_minus = ensureSize(T.mu_minus, K, d);
        T.omega1_plus = ensureSize(T.omega1_plus, K, d);
        T.omega2_plus = ensureSize(T.omega2_plus, K, d);
        T.omega1_minus = ensureSize(T.omega1_minus, K, d);
        T.omega2_minus = ensureSize(T.omega2_minus, K, d);
        P.T_log_sinusoidal = T;
    end

    if isfield(P, 'T_wavelet_mod')
        T = P.T_wavelet_mod;
        T.mu_plus = ensureSize(T.mu_plus, K, d);
        T.mu_minus = ensureSize(T.mu_minus, K, d);
        T.omega_plus = ensureSize(T.omega_plus, K, d);
        T.omega_minus = ensureSize(T.omega_minus, K, d);
        T.L_plus = ensureSize(T.L_plus, K, d);
        T.L_minus = ensureSize(T.L_minus, K, d);
        T.eta = ensureSize(T.eta, K, 1);
        P.T_wavelet_mod = T;
    end

    if isfield(P, 'T_tensor_interference')
        T = P.T_tensor_interference;
        T.mu_plus = ensureSize(T.mu_plus, K, d);
        T.mu_minus = ensureSize(T.mu_minus, K, d);
        T.omega_plus = ensureSize(T.omega_plus, K, d);
        T.omega_minus = ensureSize(T.omega_minus, K, d);
        P.T_tensor_interference = T;
    end

    if isfield(P, 'T_radial_polytrig')
        T = P.T_radial_polytrig;
        T.mu = ensureSize(T.mu, K, 1);
        T.p = ensureSize(T.p, K, 1);
        T.q = ensureSize(T.q, K, 1);
        T.omega = ensureSize(T.omega, K, 1);
        P.T_radial_polytrig = T;
    end
end

function A = ensureSize(A, rows, cols)
    % Ensure matrix A has size [rows × cols]
    % Handles cases where JSON compression changed dimensions

    if isempty(A)
        A = zeros(rows, cols);
        return;
    end

    current_size = size(A);

    % If already correct size, return as-is
    if isequal(current_size, [rows, cols])
        return;
    end

    % If scalar and should be K×1 or K×d
    if isscalar(A)
        A = repmat(A, rows, cols);
        return;
    end

    % If row vector when should be column vector
    if isequal(current_size, [1, rows]) && cols == 1
        A = A';
        return;
    end

    % If column vector when should be row vector
    if isequal(current_size, [rows, 1]) && cols > 1
        % This shouldn't happen but handle it
        A = repmat(A, 1, cols);
        return;
    end

    % If dimensions are swapped
    if isequal(current_size, [cols, rows])
        A = A';
        return;
    end

    % If it's a row vector [1×d] but should be [K×d] with K=1
    if isequal(current_size, [1, cols]) && rows == 1
        % Already correct, just ensure it's treated as 1×cols
        return;
    end

    % If total elements match, try to reshape
    if numel(A) == rows * cols
        A = reshape(A, rows, cols);
        return;
    end

    % If we get here, something unexpected happened
    warning('Cannot reshape array from [%s] to [%d×%d]', mat2str(current_size), rows, cols);
end

%% Helper function to ensure 3D array has correct dimensions
function A = ensureSize3D(A, dim1, dim2, dim3)
    % Ensure 3D array A has size [dim1 × dim2 × dim3]
    % Handles cases where JSON compression changed dimensions for K=1
    %
    % Input:
    %   A - Input array (may be 2D or 3D)
    %   dim1, dim2, dim3 - Target dimensions
    % Output:
    %   A - Reshaped array with correct dimensions

    if isempty(A)
        A = zeros(dim1, dim2, dim3);
        return;
    end

    current_size = size(A);
    current_ndims = ndims(A);

    % If already correct size
    if isequal(current_size, [dim1, dim2, dim3])
        return;
    end

    % Case 1: K=1 and array was compressed from [d×d×1] to [d×d]
    if current_ndims == 2 && dim3 == 1 && current_size(1) == dim1 && current_size(2) == dim2
        A = reshape(A, dim1, dim2, 1);
        return;
    end

    % Case 2: Array is already 3D but dimensions match by element count
    if numel(A) == dim1 * dim2 * dim3
        A = reshape(A, dim1, dim2, dim3);
        return;
    end

    % If we get here, something unexpected happened
    warning('Cannot reshape 3D array from [%s] to [%d×%d×%d]', ...
        mat2str(current_size), dim1, dim2, dim3);
end

%% Helper function to compute rotation matrix from Givens angles
function R = givens_from_psi(Psi)
    % Computes rotation matrix R from upper-triangular angle matrix Psi
    % using Givens rotations
    %
    % Input:
    %   Psi - d×d upper-triangular matrix of rotation angles (radians)
    % Output:
    %   R - d×d orthogonal rotation matrix

    d = size(Psi, 1);
    R = eye(d);
    for u = 1:d-1
        for v = u+1:d
            psi = Psi(u,v);
            if ~isfinite(psi) || abs(psi) < 1e-15
                continue;
            end
            G = eye(d);
            c = cos(psi);
            s = sin(psi);
            G([u v], [u v]) = [c -s; s c];
            R = R * G;
        end
    end
end
