%********************************** PORTAL Fitness Function **********************************
% Author: Mai Peng, Danial Yazdani, Delaram Yazdani
% Email: pengmai1998@gmail.com
%        danial.yazdani@gmail.com
%        delaram.yazdani@yahoo.com
% Last Edited: 2025-11-28
% Title: PORTAL Fitness Evaluation
% --------
% Description:
%   Evaluates the fitness of a solution vector using the PORTAL benchmark
% --------
% Inputs:
%   x - Solution vector (d×1 or 1×d)
%   PORTAL - PORTAL benchmark instance structure
% Output:
%   f - Fitness value (scalar)
%************************************************************************************************

function f = fitness(x, PORTAL)
    % Ensure x is a column vector
    if size(x, 1) == 1
        x = x';
    end

    % Check dimension
    if length(x) ~= PORTAL.d
        error('Solution dimension mismatch. Expected %d, got %d', PORTAL.d, length(x));
    end

    % Check bounds
    if any(x < PORTAL.bounds_min) || any(x > PORTAL.bounds_max)
        warning('Solution out of bounds. Clamping to feasible region.');
        x = max(PORTAL.bounds_min, min(PORTAL.bounds_max, x));
    end

    % Evaluate all components and take minimum
    K = PORTAL.K;
    vals = zeros(K, 1);

    for k = 1:K
        seq_k = seq_for_component(PORTAL.transform_seq, k, K);
        vals(k) = phi_component(x, PORTAL.P, PORTAL, k, seq_k);
    end

    f = min(vals);
end

%% ==================== Component Evaluation ====================

function f = phi_component(x, P, PORTAL, k, seq_k)
    d = PORTAL.d;
    ck = P.c(k, :)';

    % Internal coordinate transformation
    a = P.R(:,:,k) * (x - ck);

    % Apply transformation sequence
    z = apply_transform_seq(a, P, seq_k, k);

    % Compute fitness based on form
    switch P.form(k)
        case 1 % Form A
            acc = 0;
            for i = 1:d
                if z(i) >= 0
                    h = P.kappa_plus(k, i);
                    p_i = P.p_plus(k, i);
                    rho_i = P.rho_plus(k, i);
                else
                    h = P.kappa_minus(k, i);
                    p_i = P.p_minus(k, i);
                    rho_i = P.rho_minus(k, i);
                end
                acc = acc + rho_i * h * (abs(z(i))^(2 * p_i));
            end
            f = P.beta(k) + acc;

        case 2 % Form B
            qsum = 0;
            for i = 1:d
                if z(i) >= 0
                    h = P.kappa_plus(k, i);
                else
                    h = P.kappa_minus(k, i);
                end
                qsum = qsum + h^(1/max(P.p_single(k), eps)) * (abs(z(i))^2);
            end
            f = P.beta(k) + P.rho_single(k) * (qsum)^P.p_single(k);
    end
end

%% ==================== Transform Application ====================

function z = apply_transform_seq(a, P, seq, k)
    z = a;
    for t = 1:numel(seq)
        z = apply_transform_once(z, P, seq{t}, k);
    end
end

function z = apply_transform_once(a, P, name, k)
    d = numel(a);
    z = a;
    epslog = 1e-12;

    switch lower(name)
        case 'none'
            return;

        case 'additive_periodic'
            for i = 1:d
                if a(i) >= 0
                    mu = P.T_additive_periodic.mu_plus(k, i);
                    gam = P.T_additive_periodic.gamma_plus(k, i);
                    om = P.T_additive_periodic.omega_plus(k, i);
                else
                    mu = P.T_additive_periodic.mu_minus(k, i);
                    gam = P.T_additive_periodic.gamma_minus(k, i);
                    om = P.T_additive_periodic.omega_minus(k, i);
                end
                g = a(i) * (1 - exp(-gam * abs(a(i)))) * sin(om * abs(a(i)));
                z(i) = a(i) + mu * g;
            end

        case 'log_sinusoidal'
            for i = 1:d
                if a(i) == 0
                    z(i) = 0;
                    continue;
                end
                abs_ai = abs(a(i));
                L = log(abs_ai + epslog);
                if a(i) > 0
                    mu = P.T_log_sinusoidal.mu_plus(k, i);
                    o1 = P.T_log_sinusoidal.omega1_plus(k, i);
                    o2 = P.T_log_sinusoidal.omega2_plus(k, i);
                else
                    mu = P.T_log_sinusoidal.mu_minus(k, i);
                    o1 = P.T_log_sinusoidal.omega1_minus(k, i);
                    o2 = P.T_log_sinusoidal.omega2_minus(k, i);
                end
                modulation = mu * (sin(o1 * L) + sin(o2 * L));
                z(i) = sign(a(i)) * exp(L + modulation);
            end

        case 'wavelet_mod'
            eta = P.T_wavelet_mod.eta(k);
            epss = P.T_wavelet_mod.eps;
            for i = 1:d
                t = abs(a(i));
                if t < epss
                    z(i) = a(i);
                    continue;
                end
                if a(i) >= 0
                    mu = P.T_wavelet_mod.mu_plus(k, i);
                    omg = P.T_wavelet_mod.omega_plus(k, i);
                    L = P.T_wavelet_mod.L_plus(k, i);
                    if isempty(L) || L <= 0
                        L = eta / max(omg, epss);
                    end
                else
                    mu = P.T_wavelet_mod.mu_minus(k, i);
                    omg = P.T_wavelet_mod.omega_minus(k, i);
                    L = P.T_wavelet_mod.L_minus(k, i);
                    if isempty(L) || L <= 0
                        L = eta / max(omg, epss);
                    end
                end
                if omg <= 0 || mu == 0
                    z(i) = a(i);
                    continue;
                end
                env = (t/L)^2 * exp(-(t/L)^2);
                phi = env * sin(omg*t - pi/2);
                z(i) = a(i) + mu * phi * (a(i)/(t + epss));
            end

        case 'tensor_interference'
            z = a;
            for i = 1:d
                ai = a(i);
                if ai >= 0
                    mu_i = P.T_tensor_interference.mu_plus(k, i);
                    om_i = P.T_tensor_interference.omega_plus(k, i);
                else
                    mu_i = P.T_tensor_interference.mu_minus(k, i);
                    om_i = P.T_tensor_interference.omega_minus(k, i);
                end
                if mu_i == 0 || om_i <= 0
                    z(i) = ai;
                    continue;
                end

                Phi_even = 1.0;
                for j = 1:d
                    if j == i
                        continue;
                    end
                    aj = a(j);
                    if aj >= 0
                        om_j = P.T_tensor_interference.omega_plus(k, j);
                    else
                        om_j = P.T_tensor_interference.omega_minus(k, j);
                    end
                    Phi_even = Phi_even * sin(om_j * aj)^2;
                end

                scaled_mu = mu_i * (2^(d-1));
                z(i) = ai + scaled_mu * sin(om_i * ai) * Phi_even;
            end

        case 'radial_polytrig'
            mu = P.T_radial_polytrig.mu(k);
            p = P.T_radial_polytrig.p(k);
            q = P.T_radial_polytrig.q(k);
            om = P.T_radial_polytrig.omega(k);
            epss = P.T_radial_polytrig.eps;
            r = norm(a);
            if r < epss
                z = a;
                return;
            end
            phi = r^p * sin(om * r^q);
            z = a + mu * phi * (a/(r + epss));
    end
end

%% ==================== Helper Functions ====================

function seq = seq_for_component(seq_cfg, k, K)
    if ~iscell(seq_cfg) || numel(seq_cfg) ~= K
        error('transform_seq must be a 1×K cell; each entry a list of transform names.');
    end
    item = seq_cfg{k};
    if isempty(item)
        seq = {};
    elseif iscell(item)
        seq = cellfun(@char, item(:).', 'UniformOutput', false);
    elseif isstring(item)
        seq = cellstr(item);
    elseif ischar(item)
        seq = {item};
    else
        error('transform_seq{%d} must be cellstr, string, char, or empty.', k);
    end
end
