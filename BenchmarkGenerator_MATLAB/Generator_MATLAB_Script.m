%********************************** PORTAL Benchmark Generator **********************************
%Author: Generated for PORTAL Framework
%Last Edited: 2025-10-10
%Title: PORTAL Benchmark Instance Generator
% --------
% Description:
%   Generates random instances of PORTAL benchmark with configurable parameters
%   Supports saving/loading instances to/from JSON files
% --------
% License:
%   This program is to be used under the terms of the GNU General Public License
%   (http://www.gnu.org/copyleft/gpl.html).
%************************************************************************************************

function [PORTAL] = Generator_MATLAB_Script(varargin)
%% Auto-include all subdirectories
persistent paths_added;
if isempty(paths_added)
    script_path = fileparts(mfilename('fullpath'));
    addpath(genpath(script_path));
    paths_added = true;
end

%% Parse input arguments
p = inputParser;
addParameter(p, 'Seed', round(posixtime(datetime('now'))), @isnumeric);
addParameter(p, 'Dimension', 2, @isnumeric);
addParameter(p, 'NumComponents', 1, @isnumeric);

% Transformation Sequence Mode
addParameter(p, 'TransformMode', 1, @isnumeric);     % 1-4: transformation generation mode (default: Fully random)
addParameter(p, 'TransformSeq', {{}}, @iscell);  % For TransformMode=4 (user-specified)
addParameter(p, 'TransformCountMin', 0, @isnumeric); % For TransformMode=1 (random count)
addParameter(p, 'TransformCountMax', 1, @isnumeric); % For TransformMode=1 (random count)
addParameter(p, 'TransformCount', 1, @isnumeric);    % For TransformMode=2 (fixed count), scalar or K-element array

% Baseline Selection Mode
addParameter(p, 'BaselineMode', 1, @isnumeric);      % 1-4: baseline selection mode (default: random 50-50)
addParameter(p, 'BaselineProbA', 0.5, @isnumeric);   % For BaselineMode=2 (custom probability)
addParameter(p, 'BaselineType', 'A', @ischar);       % For BaselineMode=3 (all same: 'A' or 'B')
addParameter(p, 'BaselineSeq', {}, @iscell);         % For BaselineMode=4 (user-specified)

% Rotation parameters (modes 1-7)
addParameter(p, 'RotationMode', 2, @isnumeric);      % 1-7: rotation mode (default: fully connected)
addParameter(p, 'RotationProb', 0.5, @isnumeric);    % Probability for mode 3 (probabilistic)
addParameter(p, 'SpecificAngle', 45, @isnumeric);    % Fixed angle for mode 5 (degrees)
addParameter(p, 'BlockSizes', [], @isnumeric);       % Block sizes for mode 7 (block-diagonal)
addParameter(p, 'BlockAngles', [], @isnumeric);      % Block angles for mode 7 (degrees)

% Parameter sampling modes
addParameter(p, 'CenterMode', 1, @isnumeric);        % 1-3: center generation mode
addParameter(p, 'ExclusionZone', [-20, 20], @isnumeric);  % For CenterMode=2
addParameter(p, 'KappaMode', 1, @isnumeric);         % 1-5: scaling factor mode
addParameter(p, 'BetaAlpha', 0.2, @isnumeric);       % Beta distribution α for KappaMode=2
addParameter(p, 'BetaBeta', 0.2, @isnumeric);        % Beta distribution β for KappaMode=2
addParameter(p, 'PMode', 1, @isnumeric);             % 1-4: exponent mode
addParameter(p, 'FixedP', 0.8, @isnumeric);          % Fixed value for PMode=2
addParameter(p, 'PBetaAlpha', 0.5, @isnumeric);      % Beta distribution α for PMode=4
addParameter(p, 'PBetaBeta', 0.5, @isnumeric);       % Beta distribution β for PMode=4
addParameter(p, 'BetaOffsetMode', 1, @isnumeric);    % 1-4: beta offset mode
addParameter(p, 'FixedBeta', 50, @isnumeric);        % Fixed value for BetaOffsetMode=3
addParameter(p, 'BetaOffsetBetaAlpha', 0.2, @isnumeric);  % Beta distribution α for BetaOffsetMode=4
addParameter(p, 'BetaOffsetBetaBeta', 0.2, @isnumeric);   % Beta distribution β for BetaOffsetMode=4

% Transformation Parameter Sampling Mode
addParameter(p, 'TransformParamMode', 1, @isnumeric); % 1-5: transformation parameter sampling mode (default: independent)
addParameter(p, 'TransformParamBetaAlpha', 0.2, @isnumeric);  % Beta distribution α for TransformParamMode=5
addParameter(p, 'TransformParamBetaBeta', 0.2, @isnumeric);   % Beta distribution β for TransformParamMode=5

% Neutralization Mode (reserved for future)
addParameter(p, 'NeutralizationMode', 1, @isnumeric); % 1: fixed (current behavior), 2-4: reserved

parse(p, varargin{:});

%% Initialize structure
PORTAL = struct();

%% Basic Configuration
PORTAL.Seed = p.Results.Seed;
PORTAL.Rng = RandStream('mt19937ar', 'Seed', PORTAL.Seed);
PORTAL.d = p.Results.Dimension; % Dimension
PORTAL.K = p.Results.NumComponents; % Number of components

%% Domain Bounds
PORTAL.bounds_min = -100;
PORTAL.bounds_max = 100;

%% Baseline Selection (Form A or Form B for each component) - Using P1 modes
switch p.Results.BaselineMode
    case 1
        PORTAL.baseline = sample_baseline(PORTAL.Rng, PORTAL.K, 1);
    case 2
        PORTAL.baseline = sample_baseline(PORTAL.Rng, PORTAL.K, 2, p.Results.BaselineProbA);
    case 3
        PORTAL.baseline = sample_baseline(PORTAL.Rng, PORTAL.K, 3, p.Results.BaselineType);
    case 4
        PORTAL.baseline = sample_baseline(PORTAL.Rng, PORTAL.K, 4, p.Results.BaselineSeq);
    otherwise
        warning('Invalid BaselineMode %d. Using mode 1 (random 50-50).', p.Results.BaselineMode);
        PORTAL.baseline = sample_baseline(PORTAL.Rng, PORTAL.K, 1);
end

%% Transformation Sequence - Using P0 modes
switch p.Results.TransformMode
    case 1  % Fully random: random count and content
        PORTAL.transform_seq = sample_transformseq(PORTAL.Rng, PORTAL.K, 1, ...
            p.Results.TransformCountMin, p.Results.TransformCountMax);
    case 2  % Fixed count, random content
        PORTAL.transform_seq = sample_transformseq(PORTAL.Rng, PORTAL.K, 2, ...
            p.Results.TransformCount);
    case 3  % No transformations
        PORTAL.transform_seq = sample_transformseq(PORTAL.Rng, PORTAL.K, 3);
    case 4  % User-specified (backward compatibility)
        PORTAL.transform_seq = sample_transformseq(PORTAL.Rng, PORTAL.K, 4, ...
            p.Results.TransformSeq);
    otherwise
        warning('Invalid TransformMode %d. Using mode 4 (user-specified).', p.Results.TransformMode);
        PORTAL.transform_seq = sample_transformseq(PORTAL.Rng, PORTAL.K, 4, ...
            p.Results.TransformSeq);
end

%% Rotation Mode Configuration
if p.Results.RotationMode < 1 || p.Results.RotationMode > 7
    warning('RotationMode must be 1-7. Using default mode 2 (fully connected).');
    p.Results.RotationMode = 2;
end

%% ==================== Parameter Ranges (as per Table) ====================

% Component-Level Parameters
ranges.c_min = PORTAL.bounds_min + 10;  % Keep centers away from boundaries
ranges.c_max = PORTAL.bounds_max - 10;
ranges.beta_min = 0;
ranges.beta_max = 10;
ranges.p_min = 0.2;
ranges.p_max = 1.2;
ranges.kappa_min = 1;
ranges.kappa_max = 20;
ranges.theta_min = -180; % degrees
ranges.theta_max = 180;

% Neutralization Parameters
ranges.Delta_min = 100;
ranges.Delta_max = 100;
ranges.r_ref_min = 100;
ranges.r_ref_max = 100;

% Additive Periodic Transform
ranges.AP_mu_min = 0.1;
ranges.AP_mu_max = 0.7;
ranges.AP_gamma_min = 0.002;
ranges.AP_gamma_max = 0.2;
ranges.AP_omega_min = 0.05;
ranges.AP_omega_max = 1.0;

% Log-Sinusoidal Transform
ranges.LS_mu_min = 0.05;
ranges.LS_mu_max = 0.50;
ranges.LS_omega_min = 5;
ranges.LS_omega_max = 50;

% Wavelet-Inspired Transform
ranges.WM_mu_min = 10;
ranges.WM_mu_max = 50;
ranges.WM_omega_min = 0.3;
ranges.WM_omega_max = 1.0;
ranges.WM_L_min = 10;
ranges.WM_L_max = 80;
ranges.WM_eta_min = 10;
ranges.WM_eta_max = 24;

% Tensor Interference Transform
ranges.TI_mu0_min = 10;
ranges.TI_mu0_max = 20;
ranges.TI_omega_min = 0.1;
ranges.TI_omega_max = 0.7;

% Radial Polytrig Transform
ranges.RP_mu_min = 0.4;
ranges.RP_mu_max = 2.0;
ranges.RP_p_min = 0.4;
ranges.RP_p_max = 0.7;
ranges.RP_q_min = 0.4;
ranges.RP_q_max = 1.2;
ranges.RP_omega_min = 0.1;
ranges.RP_omega_max = 10;

PORTAL.ranges = ranges;

%% ==================== Generate Core Parameters ====================
P = struct();

% Centers c_k (K×d) - Using sampling function
if p.Results.CenterMode == 2
    P.c = sample_centers(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, ...
                        p.Results.CenterMode, p.Results.ExclusionZone);
else
    P.c = sample_centers(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, p.Results.CenterMode);
end

% Offsets β_k (K×1) - Using sampling function
if p.Results.BetaOffsetMode == 3
    P.beta = sample_beta(PORTAL.Rng, PORTAL.K, ranges, ...
                        p.Results.BetaOffsetMode, p.Results.FixedBeta);
elseif p.Results.BetaOffsetMode == 4
    beta_params = [p.Results.BetaOffsetBetaAlpha, p.Results.BetaOffsetBetaBeta];
    P.beta = sample_beta(PORTAL.Rng, PORTAL.K, ranges, ...
                        p.Results.BetaOffsetMode, beta_params);
else
    P.beta = sample_beta(PORTAL.Rng, PORTAL.K, ranges, p.Results.BetaOffsetMode);
end

% Scaling factors κ_{k,i}^{±} (K×d) - Using sampling function
if p.Results.KappaMode == 2
    beta_params = [p.Results.BetaAlpha, p.Results.BetaBeta];
    P.kappa_plus = sample_kappa(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, ...
                                p.Results.KappaMode, beta_params);
    P.kappa_minus = sample_kappa(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, ...
                                 p.Results.KappaMode, beta_params);
else
    P.kappa_plus = sample_kappa(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, p.Results.KappaMode);
    P.kappa_minus = sample_kappa(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, p.Results.KappaMode);
end

% Exponents - Using sampling function
if p.Results.PMode == 2
    P.p_single = sample_exponents(PORTAL.Rng, PORTAL.K, 1, ranges, ...
                                 p.Results.PMode, p.Results.FixedP);
    P.p_plus = sample_exponents(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, ...
                               p.Results.PMode, p.Results.FixedP);
    P.p_minus = sample_exponents(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, ...
                                p.Results.PMode, p.Results.FixedP);
elseif p.Results.PMode == 4
    beta_params = [p.Results.PBetaAlpha, p.Results.PBetaBeta];
    P.p_single = sample_exponents(PORTAL.Rng, PORTAL.K, 1, ranges, ...
                                 p.Results.PMode, beta_params);
    P.p_plus = sample_exponents(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, ...
                               p.Results.PMode, beta_params);
    P.p_minus = sample_exponents(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, ...
                                p.Results.PMode, beta_params);
else
    P.p_single = sample_exponents(PORTAL.Rng, PORTAL.K, 1, ranges, p.Results.PMode);
    P.p_plus = sample_exponents(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, p.Results.PMode);
    P.p_minus = sample_exponents(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges, p.Results.PMode);
end

% Rotation angles via Ψ (Givens angles) - store angles in degrees
% Supports 7 rotation modes for different interaction patterns
P.Psi_deg = zeros(PORTAL.d, PORTAL.d, PORTAL.K); % Store angles in degrees
P.R = zeros(PORTAL.d, PORTAL.d, PORTAL.K); % Compute R for immediate use

for k = 1:PORTAL.K
    Psi_deg_k = zeros(PORTAL.d);

    switch p.Results.RotationMode
        case 1 % No rotation (identity matrix)
            % Psi_deg_k remains zeros, R will be identity matrix

        case 2 % Fully connected random rotations (all pairs have angles)
            for u = 1:PORTAL.d-1
                for v = u+1:PORTAL.d
                    Psi_deg_k(u,v) = ranges.theta_min + ...
                        (ranges.theta_max - ranges.theta_min) * rand(PORTAL.Rng);
                end
            end

        case 3 % Probabilistic connections (sparse interactions)
            prob = p.Results.RotationProb;
            for u = 1:PORTAL.d-1
                for v = u+1:PORTAL.d
                    if rand(PORTAL.Rng) < prob
                        Psi_deg_k(u,v) = ranges.theta_min + ...
                            (ranges.theta_max - ranges.theta_min) * rand(PORTAL.Rng);
                    end
                end
            end

        case 4 % Uniform angle per component (same angle for all pairs)
            angle = ranges.theta_min + (ranges.theta_max - ranges.theta_min) * rand(PORTAL.Rng);
            for u = 1:PORTAL.d-1
                for v = u+1:PORTAL.d
                    Psi_deg_k(u,v) = angle;
                end
            end

        case 5 % Fixed angle (user-specified angle for all pairs)
            specific_angle = p.Results.SpecificAngle;
            for u = 1:PORTAL.d-1
                for v = u+1:PORTAL.d
                    Psi_deg_k(u,v) = specific_angle;
                end
            end

        case 6 % Chain-like structure (nearest neighbor connections only)
            % Only connect adjacent dimensions: (1,2), (2,3), ..., (d-1,d)
            for u = 1:PORTAL.d-1
                Psi_deg_k(u, u+1) = ranges.theta_min + ...
                    (ranges.theta_max - ranges.theta_min) * rand(PORTAL.Rng);
            end

        case 7 % Block-diagonal (partially separable groups)
            % Determine block structure
            if isempty(p.Results.BlockSizes)
                % Default: create balanced blocks
                if PORTAL.d >= 30
                    n_blocks = 3;
                else
                    n_blocks = 2;
                end
                base_size = floor(PORTAL.d / n_blocks);
                remainder = mod(PORTAL.d, n_blocks);
                block_sizes = repmat(base_size, 1, n_blocks);
                for b = 1:remainder
                    block_sizes(b) = block_sizes(b) + 1;
                end
            else
                block_sizes = p.Results.BlockSizes;
            end

            % Set block angles
            if isempty(p.Results.BlockAngles) || length(p.Results.BlockAngles) ~= length(block_sizes)
                % Default angles for each block
                default_angles = [45, 135, 22.5, 112.5];
                block_angles = default_angles(1:length(block_sizes));
            else
                block_angles = p.Results.BlockAngles;
            end

            % Verify total size
            if sum(block_sizes) > PORTAL.d
                error('Sum of block sizes (%d) exceeds dimension (%d)', sum(block_sizes), PORTAL.d);
            end

            % Assign dimensions to blocks randomly
            all_dims = randperm(PORTAL.Rng, PORTAL.d);
            dim_start = 1;

            for b = 1:length(block_sizes)
                dim_end = dim_start + block_sizes(b) - 1;
                block_dims = all_dims(dim_start:dim_end);

                % Apply rotation within block only
                for i = 1:length(block_dims)
                    for j = i+1:length(block_dims)
                        u = block_dims(i);
                        v = block_dims(j);
                        if u > v
                            temp = u; u = v; v = temp;
                        end
                        Psi_deg_k(u, v) = block_angles(b);
                    end
                end
                dim_start = dim_end + 1;
            end

        otherwise
            % Should not reach here due to validation above
            warning('Unknown rotation mode %d, using identity matrix', p.Results.RotationMode);
    end

    P.Psi_deg(:,:,k) = Psi_deg_k; % Store degrees
    P.R(:,:,k) = givens_from_psi(deg2rad(Psi_deg_k)); % Convert to radians for R
end

% Neutralization parameters
P.neutralize_enable = true;
P.Delta = ranges.Delta_min + (ranges.Delta_max - ranges.Delta_min) * rand(PORTAL.Rng, PORTAL.K, 1);
P.r_ref = ranges.r_ref_min + (ranges.r_ref_max - ranges.r_ref_min) * rand(PORTAL.Rng, PORTAL.K, 1);

PORTAL.P = P;

%% ==================== Generate Transform Parameters ====================
% Using P2 - TransformParamMode for unified parameter sampling

tpm = p.Results.TransformParamMode;  % Shorthand for transform parameter mode

% Prepare extra args for mode 5 (Beta distribution)
if tpm == 5
    tpm_extra_args = {[p.Results.TransformParamBetaAlpha, p.Results.TransformParamBetaBeta]};
else
    tpm_extra_args = {};
end

% Additive Periodic
T_AP = struct();
T_AP.mu_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.AP_mu_min, ranges.AP_mu_max, tpm, tpm_extra_args{:});
T_AP.mu_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.AP_mu_min, ranges.AP_mu_max, tpm, tpm_extra_args{:});
T_AP.gamma_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.AP_gamma_min, ranges.AP_gamma_max, tpm, tpm_extra_args{:});
T_AP.gamma_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.AP_gamma_min, ranges.AP_gamma_max, tpm, tpm_extra_args{:});
T_AP.omega_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.AP_omega_min, ranges.AP_omega_max, tpm, tpm_extra_args{:});
T_AP.omega_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.AP_omega_min, ranges.AP_omega_max, tpm, tpm_extra_args{:});
PORTAL.P.T_additive_periodic = T_AP;

% Log-Sinusoidal
T_LS = struct();
T_LS.mu_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.LS_mu_min, ranges.LS_mu_max, tpm, tpm_extra_args{:});
T_LS.mu_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.LS_mu_min, ranges.LS_mu_max, tpm, tpm_extra_args{:});
T_LS.omega1_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.LS_omega_min, ranges.LS_omega_max, tpm, tpm_extra_args{:});
T_LS.omega2_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.LS_omega_min, ranges.LS_omega_max, tpm, tpm_extra_args{:});
T_LS.omega1_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.LS_omega_min, ranges.LS_omega_max, tpm, tpm_extra_args{:});
T_LS.omega2_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.LS_omega_min, ranges.LS_omega_max, tpm, tpm_extra_args{:});
PORTAL.P.T_log_sinusoidal = T_LS;

% Wavelet-Inspired
T_WM = struct();
T_WM.mu_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.WM_mu_min, ranges.WM_mu_max, tpm, tpm_extra_args{:});
T_WM.mu_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.WM_mu_min, ranges.WM_mu_max, tpm, tpm_extra_args{:});
T_WM.omega_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.WM_omega_min, ranges.WM_omega_max, tpm, tpm_extra_args{:});
T_WM.omega_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.WM_omega_min, ranges.WM_omega_max, tpm, tpm_extra_args{:});
T_WM.L_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.WM_L_min, ranges.WM_L_max, tpm, tpm_extra_args{:});
T_WM.L_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.WM_L_min, ranges.WM_L_max, tpm, tpm_extra_args{:});
T_WM.eta = ranges.WM_eta_min + (ranges.WM_eta_max - ranges.WM_eta_min) * rand(PORTAL.Rng, PORTAL.K, 1);
T_WM.eps = 1e-12;
PORTAL.P.T_wavelet_mod = T_WM;

% Tensor Interference
T_TI = struct();
T_TI.mu_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.TI_mu0_min, ranges.TI_mu0_max, tpm, tpm_extra_args{:});
T_TI.mu_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.TI_mu0_min, ranges.TI_mu0_max, tpm, tpm_extra_args{:});
T_TI.omega_plus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.TI_omega_min, ranges.TI_omega_max, tpm, tpm_extra_args{:});
T_TI.omega_minus = sample_transform_param(PORTAL.Rng, PORTAL.K, PORTAL.d, ranges.TI_omega_min, ranges.TI_omega_max, tpm, tpm_extra_args{:});
PORTAL.P.T_tensor_interference = T_TI;

% Radial Polytrig (note: all parameters are K×1, not K×d)
T_RP = struct();
T_RP.mu = ranges.RP_mu_min + (ranges.RP_mu_max - ranges.RP_mu_min) * rand(PORTAL.Rng, PORTAL.K, 1);
T_RP.p = ranges.RP_p_min + (ranges.RP_p_max - ranges.RP_p_min) * rand(PORTAL.Rng, PORTAL.K, 1);
T_RP.q = ranges.RP_q_min + (ranges.RP_q_max - ranges.RP_q_min) * rand(PORTAL.Rng, PORTAL.K, 1);
T_RP.omega = ranges.RP_omega_min + (ranges.RP_omega_max - ranges.RP_omega_min) * rand(PORTAL.Rng, PORTAL.K, 1);
T_RP.eps = 1e-12;
PORTAL.P.T_radial_polytrig = T_RP;

%% ==================== Precompute Scalings ====================
[PORTAL.P.form, PORTAL.P.rho_single, PORTAL.P.rho_plus, PORTAL.P.rho_minus] = ...
    precompute_scalings(PORTAL);

fprintf('PORTAL Benchmark Instance Generated:\n');
fprintf('  Dimension: %d\n', PORTAL.d);
fprintf('  Components: %d\n', PORTAL.K);
fprintf('  Seed: %d\n', PORTAL.Seed);

% Display rotation mode with descriptive name
rotation_names = {'No Rotation', 'Fully Connected', 'Probabilistic', ...
                  'Uniform Angle', 'Fixed Angle', 'Chain-like', 'Block-diagonal'};
fprintf('  Rotation Mode: %d (%s)\n', p.Results.RotationMode, rotation_names{p.Results.RotationMode});

% Display sampling modes
fprintf('  Center Mode: %d, Kappa Mode: %d, P Mode: %d\n', ...
        p.Results.CenterMode, p.Results.KappaMode, p.Results.PMode);

fprintf('  Baselines: ');
for k = 1:PORTAL.K
    fprintf('%s ', PORTAL.baseline{k});
end
fprintf('\n');

end

%% ==================== Helper Functions ====================

function R = givens_from_psi(Psi)
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

function [form, rho_single, rho_plus, rho_minus] = precompute_scalings(PORTAL)
    K = PORTAL.K;
    d = PORTAL.d;
    P = PORTAL.P;

    form = zeros(K, 1);
    rho_single = NaN(K, 1);
    rho_plus = NaN(K, d);
    rho_minus = NaN(K, d);

    for k = 1:K
        % Determine form based on baseline
        if strcmp(PORTAL.baseline{k}, 'A')
            form(k) = 1; % Form A
        else
            form(k) = 2; % Form B
        end

        if P.neutralize_enable
            switch form(k)
                case 1 % Form A
                    bar_kappa_ki = (P.kappa_plus(k,:) + P.kappa_minus(k,:)) / 2;
                    bar_kappa_k = mean(bar_kappa_ki);
                    for i = 1:d
                        rho_plus(k,i) = (P.Delta(k)/d) / (bar_kappa_k * (P.r_ref(k))^(2 * P.p_plus(k,i)));
                        rho_minus(k,i) = (P.Delta(k)/d) / (bar_kappa_k * (P.r_ref(k))^(2 * P.p_minus(k,i)));
                    end
                case 2 % Form B
                    bar_kappa_ki = (P.kappa_plus(k,:) + P.kappa_minus(k,:)) / 2;
                    p = P.p_single(k);
                    sum_term = sum(bar_kappa_ki .^ (1/max(p, eps)));
                    rho_single(k) = P.Delta(k) / ((P.r_ref(k)^2 * sum_term) ^ max(p, eps));
            end
        else
            switch form(k)
                case 1
                    rho_plus(k,:) = 1;
                    rho_minus(k,:) = 1;
                case 2
                    rho_single(k) = 1;
            end
        end
    end
end

%% ==================== Parameter Sampling Functions ====================

function c = sample_centers(rng_stream, K, d, ranges, mode, varargin)
%SAMPLE_CENTERS Sample component center positions with different strategies
    switch mode
        case 1  % Uniform random (current default behavior)
            c = ranges.c_min + (ranges.c_max - ranges.c_min) * rand(rng_stream, K, d);

        case 2  % With exclusion zone
            if isempty(varargin)
                error('Mode 2 requires exclusion_zone parameter [lower, upper]');
            end
            exclusion_zone = varargin{1};

            % Generate candidates in lower or upper ranges
            lower_range = ranges.c_min + ...
                         (exclusion_zone(1) - ranges.c_min) * ...
                         rand(rng_stream, K, d);
            upper_range = exclusion_zone(2) + ...
                         (ranges.c_max - exclusion_zone(2)) * ...
                         rand(rng_stream, K, d);

            % Randomly select from lower or upper range for each component
            selector = randi(rng_stream, [0, 1], K, d);
            c = selector .* lower_range + (1 - selector) .* upper_range;

        case 3  % Shared center (all components overlapping)
            % Generate one random center and replicate for all components
            center = ranges.c_min + (ranges.c_max - ranges.c_min) * rand(rng_stream, 1, d);
            c = repmat(center, K, 1);

        otherwise
            error('Invalid center_mode: %d. Valid modes are 1-3.', mode);
    end
end

function kappa = sample_kappa(rng_stream, K, d, ranges, mode, varargin)
%SAMPLE_KAPPA Sample scaling factors with different strategies
    switch mode
        case 1  % Independent random (current default)
            kappa = ranges.kappa_min + (ranges.kappa_max - ranges.kappa_min) * ...
                    rand(rng_stream, K, d);

        case 2  % Beta distribution for extreme condition numbers
            if isempty(varargin) || length(varargin{1}) < 2
                error('Mode 2 requires beta_params = [alpha, beta]');
            end
            beta_params = varargin{1};
            alpha = beta_params(1);
            beta_val = beta_params(2);

            % Sample from Beta distribution
            % Temporarily set global stream to our RandStream
            prev_stream = RandStream.getGlobalStream();
            RandStream.setGlobalStream(rng_stream);
            kappa = ranges.kappa_min + (ranges.kappa_max - ranges.kappa_min) * ...
                    betarnd(alpha, beta_val, K, d);
            RandStream.setGlobalStream(prev_stream)

        case 3  % Component-shared (isotropic scaling per component)
            scales = ranges.kappa_min + (ranges.kappa_max - ranges.kappa_min) * ...
                     rand(rng_stream, K, 1);
            kappa = repmat(scales, 1, d);

        case 4  % Linear distribution with random permutation
            kappa = zeros(K, d);
            kappa_vals = linspace(ranges.kappa_min, ranges.kappa_max, d);
            for k = 1:K
                kappa(k, :) = kappa_vals(randperm(rng_stream, d));
            end
        case 5  % isotropic for each direction
            kappa = ones(K, d);
        otherwise
            error('Invalid kappa_mode: %d. Valid modes are 1-5.', mode);
    end
end

function p_vals = sample_exponents(rng_stream, K, d, ranges, mode, varargin)
%SAMPLE_EXPONENTS Sample exponent values with different strategies
    switch mode
        case 1  % Random range (current default)
            p_vals = ranges.p_min + (ranges.p_max - ranges.p_min) * ...
                     rand(rng_stream, K, d);

        case 2  % Fixed value
            if isempty(varargin)
                error('Mode 2 requires fixed_value parameter');
            end
            fixed_value = varargin{1};
            p_vals = fixed_value * ones(K, d);

        case 3  % Gradient across dimensions
            gradient = linspace(ranges.p_min, ranges.p_max, d);
            p_vals = repmat(gradient, K, 1);

        case 4  % Beta distribution for extreme exponents
            if isempty(varargin) || length(varargin{1}) < 2
                error('Mode 4 requires beta_params = [alpha, beta]');
            end
            beta_params = varargin{1};
            alpha = beta_params(1);
            beta_val = beta_params(2);

            % Sample from Beta distribution
            % Temporarily set global stream to our RandStream
            prev_stream = RandStream.getGlobalStream();
            RandStream.setGlobalStream(rng_stream);
            p_vals = ranges.p_min + (ranges.p_max - ranges.p_min) * ...
                     betarnd(alpha, beta_val, K, d);
            RandStream.setGlobalStream(prev_stream);

        case 5  % Linear distribution with random permutation
            if d ~= 1
                p_vals = zeros(K, d);
                p_vals_linear = linspace(ranges.p_min, ranges.p_max, d);
                for k = 1:K
                    p_vals(k, :) = p_vals_linear(randperm(rng_stream, d));
                end
            else
                p_vals = ranges.p_min + (ranges.p_max - ranges.p_min) * ...
                         rand(rng_stream, K, d);                
            end

        otherwise
            error('Invalid p_mode: %d. Valid modes are 1-5.', mode);
    end
end

function beta_val = sample_beta(rng_stream, K, ranges, mode, varargin)
%SAMPLE_BETA Sample offset values with different strategies
    switch mode
        case 1  % Independent random (current default)
            beta_val = ranges.beta_min + (ranges.beta_max - ranges.beta_min) * ...
                       rand(rng_stream, K, 1);

        case 2  % Shared random value
            val = ranges.beta_min + (ranges.beta_max - ranges.beta_min) * ...
                  rand(rng_stream, 1, 1);
            beta_val = repmat(val, K, 1);

        case 3  % Fixed value
            if isempty(varargin)
                error('Mode 3 requires fixed_value parameter');
            end
            fixed_value = varargin{1};
            beta_val = fixed_value * ones(K, 1);

        case 4  % Beta distribution for extreme offsets
            if isempty(varargin) || length(varargin{1}) < 2
                error('Mode 4 requires beta_params = [alpha, beta]');
            end
            beta_params = varargin{1};
            alpha = beta_params(1);
            beta_param = beta_params(2);

            % Sample from Beta distribution
            % Temporarily set global stream to our RandStream
            prev_stream = RandStream.getGlobalStream();
            RandStream.setGlobalStream(rng_stream);
            beta_val = ranges.beta_min + (ranges.beta_max - ranges.beta_min) * ...
                       betarnd(alpha, beta_param, K, 1);
            RandStream.setGlobalStream(prev_stream);

        case 5  % Linear distribution with random permutation
            beta_vals_linear = linspace(ranges.beta_min, ranges.beta_max, K);
            beta_val = beta_vals_linear(randperm(rng_stream, K))';

        otherwise
            error('Invalid beta_mode: %d. Valid modes are 1-5.', mode);
    end
end

function transform_seq = sample_transformseq(rng_stream, K, mode, varargin)
%SAMPLE_TRANSFORMSEQ Sample transformation sequences with different strategies
%   Available transformations: additive_periodic, log_sinusoidal, wavelet_mod,
%                               tensor_interference, radial_polytrig

    available_transforms = {'additive_periodic', 'log_sinusoidal', 'wavelet_mod', ...
                           'tensor_interference', 'radial_polytrig'};

    switch mode
        case 1  % Fully random: random count per component, random transformations
            if length(varargin) < 2
                error('Mode 1 requires count_min and count_max parameters');
            end
            count_min = varargin{1};
            count_max = varargin{2};

            transform_seq = cell(K, 1);
            for k = 1:K
                % Random count for this component
                count = randi(rng_stream, [count_min, count_max]);
                if count == 0
                    transform_seq{k} = {};
                else
                    % Randomly select transformations (with repetition allowed)
                    indices = randi(rng_stream, length(available_transforms), 1, count);
                    transform_seq{k} = available_transforms(indices);
                end
            end

        case 2  % Fixed count, random content
            if isempty(varargin)
                error('Mode 2 requires count parameter (scalar or K-element array)');
            end
            count = varargin{1};

            if isscalar(count)
                counts = repmat(count, 1, K);
            elseif length(count) == K
                counts = count;
            else
                error('count must be scalar or K-element array (K=%d)', K);
            end

            transform_seq = cell(K, 1);
            for k = 1:K
                if counts(k) == 0
                    transform_seq{k} = {};
                else
                    indices = randi(rng_stream, length(available_transforms), 1, counts(k));
                    transform_seq{k} = available_transforms(indices);
                end
            end

        case 3  % No transformations (empty for all components)
            transform_seq = cell(K, 1);
            for k = 1:K
                transform_seq{k} = {};
            end

        case 4  % User-specified (backward compatibility)
            if isempty(varargin)
                error('Mode 4 requires user_seq parameter');
            end
            user_seq = varargin{1};

            if numel(user_seq) == 1 && K > 1
                % Replicate for all components
                transform_seq = repmat(user_seq, 1, K);
            else
                transform_seq = user_seq;
            end

        otherwise
            error('Invalid transform_mode: %d. Valid modes are 1-4.', mode);
    end
end

function baseline_seq = sample_baseline(rng_stream, K, mode, varargin)
%SAMPLE_BASELINE Sample baseline (Form A or B) with different strategies
    switch mode
        case 1  % Random 50-50 (current default)
            baseline_seq = cell(K, 1);
            for k = 1:K
                if rand(rng_stream) < 0.5
                    baseline_seq{k} = 'A';
                else
                    baseline_seq{k} = 'B';
                end
            end

        case 2  % Custom probability for Form A
            if isempty(varargin)
                error('Mode 2 requires prob_A parameter');
            end
            prob_A = varargin{1};

            baseline_seq = cell(K, 1);
            for k = 1:K
                if rand(rng_stream) < prob_A
                    baseline_seq{k} = 'A';
                else
                    baseline_seq{k} = 'B';
                end
            end

        case 3  % All same (all A or all B)
            if isempty(varargin)
                error('Mode 3 requires baseline_type parameter (''A'' or ''B'')');
            end
            baseline_type = varargin{1};

            if ~ismember(baseline_type, {'A', 'B'})
                error('baseline_type must be ''A'' or ''B''');
            end

            baseline_seq = cell(K, 1);
            for k = 1:K
                baseline_seq{k} = baseline_type;
            end

        case 4  % User-specified
            if isempty(varargin)
                error('Mode 4 requires user_seq parameter');
            end
            baseline_seq = varargin{1};

            if length(baseline_seq) ~= K
                error('User-specified baseline_seq must have K=%d elements', K);
            end

        otherwise
            error('Invalid baseline_mode: %d. Valid modes are 1-4.', mode);
    end
end

function param_matrix = sample_transform_param(rng_stream, K, d, range_min, range_max, mode, varargin)
%SAMPLE_TRANSFORM_PARAM Sample transformation parameter matrix with different strategies
%   Unified function for all 27 transformation parameters
    switch mode
        case 1  % Independent random (K×d, all different) - current default
            param_matrix = range_min + (range_max - range_min) * rand(rng_stream, K, d);

        case 2  % Component-shared (same across K, different across d)
            row = range_min + (range_max - range_min) * rand(rng_stream, 1, d);
            param_matrix = repmat(row, K, 1);

        case 3  % Dimension-shared (different across K, same across d)
            col = range_min + (range_max - range_min) * rand(rng_stream, K, 1);
            param_matrix = repmat(col, 1, d);

        case 4  % Global shared (all same)
            val = range_min + (range_max - range_min) * rand(rng_stream, 1, 1);
            param_matrix = val * ones(K, d);

        case 5  % Beta distribution for extreme values
            if isempty(varargin) || length(varargin{1}) < 2
                error('Mode 5 requires beta_params = [alpha, beta]');
            end
            beta_params = varargin{1};
            alpha = beta_params(1);
            beta_val = beta_params(2);

            % Sample from Beta distribution
            % Temporarily set global stream to our RandStream
            prev_stream = RandStream.getGlobalStream();
            RandStream.setGlobalStream(rng_stream);
            param_matrix = range_min + (range_max - range_min) * ...
                          betarnd(alpha, beta_val, K, d);
            RandStream.setGlobalStream(prev_stream);

        case 6  % Linear distribution with random permutation
            param_matrix = zeros(K, d);
            param_vals_linear = linspace(range_min, range_max, d);
            for k = 1:K
                param_matrix(k, :) = param_vals_linear(randperm(rng_stream, d));
            end

        otherwise
            error('Invalid transform_param_mode: %d. Valid modes are 1-6.', mode);
    end
end

function PrintModeGuide()
%PRINTMODEGUIDE Display comprehensive guide for all parameter modes
%
%   Call this function to see detailed explanations of all available modes

    fprintf('\n=== PORTAL Parameter Mode Guide ===\n\n');

    fprintf('--- Baseline Mode (BaselineMode) ---\n');
    fprintf('  1: Random 50-50 - each component has 50%% probability of Form A or Form B\n');
    fprintf('  2: Custom probability - use BaselineProbA to control probability of Form A\n');
    fprintf('  3: All same - all components use same baseline (use BaselineType: ''A'' or ''B'')\n');
    fprintf('  4: User-specified - use BaselineSeq cell array with custom sequence\n\n');

    fprintf('--- Transform Mode (TransformMode) ---\n');
    fprintf('  1: Fully random - random count per component in [TransformCountMin, TransformCountMax], random transformations\n');
    fprintf('  2: Fixed count - each component has TransformCount random transformations\n');
    fprintf('  3: No transformations - all components have empty transformation sequences\n');
    fprintf('  4: User-specified - use TransformSeq cell array (backward compatible)\n\n');

    fprintf('--- Transform Parameter Mode (TransformParamMode) ---\n');
    fprintf('  1: Independent random - each (k,i) parameter sampled independently (K×d matrix)\n');
    fprintf('  2: Component-shared - all components share same pattern across dimensions (1×d replicated)\n');
    fprintf('  3: Dimension-shared - all dimensions share same value per component (K×1 replicated)\n');
    fprintf('  4: Global shared - all components and dimensions use same value (scalar)\n');
    fprintf('  5: Beta distribution - creates extreme parameter values (use TransformParamBetaAlpha, TransformParamBetaBeta)\n');
    fprintf('  6: Linear distribution - linearly spaced values with random permutation per component\n\n');

    fprintf('--- Neutralization Mode (NeutralizationMode) ---\n');
    fprintf('  1: Fixed (current implementation) - reserved interface for future Delta/r_ref parameter control\n\n');

    fprintf('--- Center Mode (CenterMode) ---\n');
    fprintf('  1: Uniform random - centers distributed uniformly in allowed range\n');
    fprintf('  2: Exclusion zone - centers avoid specified region (use ExclusionZone)\n');
    fprintf('  3: Shared center - all components overlap at same center\n\n');

    fprintf('--- Kappa Mode (KappaMode) ---\n');
    fprintf('  1: Independent random - each scaling factor sampled independently\n');
    fprintf('  2: Beta distribution - creates extreme condition numbers (use BetaAlpha, BetaBeta)\n');
    fprintf('  3: Component-shared - isotropic scaling within each component\n');
    fprintf('  4: Linear distribution - linearly spaced values with random permutation\n');
    fprintf('  5: Isotropic (all directions) - same scaling factor used in all directions, ensuring isotropy across all dimensions\n\n');

    fprintf('--- Exponent Mode (PMode) ---\n');
    fprintf('  1: Random range - uniformly random exponents\n');
    fprintf('  2: Fixed value - all exponents set to FixedP\n');
    fprintf('  3: Gradient - linearly varying from p_min to p_max\n');
    fprintf('  4: Beta distribution - creates extreme exponent values (use PBetaAlpha, PBetaBeta)\n');
    fprintf('  5: Linear distribution - linearly spaced values with random permutation per component (Only Form A)\n\n');

    fprintf('--- Beta Offset Mode (BetaOffsetMode) ---\n');
    fprintf('  1: Independent random - each component has independent offset\n');
    fprintf('  2: Shared value - all components share same random offset\n');
    fprintf('  3: Fixed value - all components use FixedBeta\n');
    fprintf('  4: Beta distribution - creates extreme offset values (use BetaOffsetBetaAlpha, BetaOffsetBetaBeta)\n');
    fprintf('  5: Linear distribution - linearly spaced values with random permutation across components\n\n');

    fprintf('--- Rotation Mode (RotationMode) ---\n');
    fprintf('  1: No rotation - separable problem (identity matrix)\n');
    fprintf('  2: Fully connected - all variable pairs interact\n');
    fprintf('  3: Probabilistic - sparse interactions (use RotationProb)\n');
    fprintf('  4: Uniform angle - same angle for all pairs per component\n');
    fprintf('  5: Fixed angle - user-specified angle for all pairs (use SpecificAngle)\n');
    fprintf('  6: Chain-like - only adjacent dimensions interact\n');
    fprintf('  7: Block-diagonal - partially separable groups (use BlockSizes, BlockAngles)\n\n');
end
