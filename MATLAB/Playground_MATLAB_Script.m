%********************************** PORTAL Playground Function **********************************
% Author: Mai Peng, Danial Yazdani, Delaram Yazdani
% Email: pengmai1998@gmail.com
%        danial.yazdani@gmail.com
%        delaram.yazdani@yahoo.com
% Last Edited: 2025-11-28
% Title: PORTAL Landscape Visualization
% --------
% Description:
%   Visualizes the PORTAL benchmark landscape with surface, contour, and combined plots
% --------
% Inputs:
%   PORTAL - PORTAL benchmark instance structure
%   resolution - (optional) Grid resolution for plotting (default: 250)
%   show_contour - (optional) Show contour lines on surface plot (default: true)
%************************************************************************************************

function Playground_MATLAB_Script(PORTAL, resolution, show_contour)
    %% Parse input arguments
    if nargin < 2
        resolution = 250;
    end
    if nargin < 3
        show_contour = true;
    end

    %% Check dimension
    if PORTAL.d ~= 2
        warning('Visualization requires d=2. Current dimension is %d. Skipping plot.', PORTAL.d);
        return;
    end

    %% Grid evaluation
    xb = [PORTAL.bounds_min, PORTAL.bounds_max];
    [x1, x2] = meshgrid(linspace(xb(1), xb(2), resolution), ...
                        linspace(xb(1), xb(2), resolution));
    F = zeros(size(x1));

    fprintf('Evaluating landscape on %dx%d grid...\n', resolution, resolution);
    for ii = 1:numel(x1)
        x = [x1(ii); x2(ii)];
        F(ii) = fitness(x, PORTAL);
    end

    % Replace grid points closest to component centers with exact values
    for k = 1:PORTAL.K
        dist_sq = (x1 - PORTAL.P.c(k,1)).^2 + (x2 - PORTAL.P.c(k,2)).^2;
        [~, min_idx] = min(dist_sq(:));
        x1(min_idx) = PORTAL.P.c(k,1);
        x2(min_idx) = PORTAL.P.c(k,2);
        F(min_idx) = PORTAL.P.beta(k);
    end

    %% Find min/max for contour placement
    minF = min(F(:));
    maxF = max(F(:));

    fprintf('Fitness range: [%.4f, %.4f]\n', minF, maxF);
    fprintf('Generating plots...\n');

    %% Figure positioning
    side = 760;
    gap = 40;
    left1 = 100;
    top = 80;
    left2 = left1 + side + gap;
    left3 = left2 + side + gap;

    %% ---- SURFACE PLOT ----
    fig1 = figure('Name', 'PORTAL Surface', 'Position', [left1, top, side, side], 'Color', 'w');
    ax1 = axes('Parent', fig1);
    surf(ax1, x1, x2, F, 'FaceAlpha', 0.8, 'EdgeColor', 'none');
    hold(ax1, 'on');

    if show_contour
        contour3(ax1, x1, x2, F, 40, 'k', 'LineWidth', 0.6);
    end

    xlabel(ax1, 'x_1');
    ylabel(ax1, 'x_2');
    zlabel(ax1, 'f(x_1, x_2)');
    title(ax1, sprintf('PORTAL Surface (K=%d, Seed=%d)', PORTAL.K, PORTAL.Seed));
    colormap(ax1, parula);
    grid(ax1, 'on');
    view(ax1, -35, 30);
    box(ax1, 'on');
    axis(ax1, 'tight');
    pbaspect(ax1, [1 1 1]);
    square_axes_fill(ax1);
    rotate3d(ax1, 'on');

    %% ---- CONTOUR PLOT WITH COLORBAR ----
    fig2 = figure('Name', 'PORTAL Contour', 'Position', [left2, top, side, side], 'Color', 'w');
    ax2 = axes('Parent', fig2);
    contour(ax2, x1, x2, F, 40, 'LineWidth', 1);
    hold(ax2, 'on');

    % Mark component centers
    for k = 1:PORTAL.K
        plot(ax2, PORTAL.P.c(k,1), PORTAL.P.c(k,2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
    end

    axis(ax2, 'equal');
    axis(ax2, 'tight');
    grid(ax2, 'on');
    xlabel(ax2, 'x_1');
    ylabel(ax2, 'x_2');
    title(ax2, sprintf('PORTAL Contour (K=%d, Seed=%d)', PORTAL.K, PORTAL.Seed));
    colormap(ax2, parula);
    cb = colorbar(ax2);
    ylabel(cb, 'f(x_1, x_2)');
    box(ax2, 'on');
    square_axes_fill_with_colorbar(ax2);

    %% ---- COMBINED SURFACE + BASE CONTOUR ----
    fig3 = figure('Name', 'PORTAL Combined', 'Position', [left3, top, side, side], 'Color', 'w');
    ax3 = axes('Parent', fig3);

    % Plot the surface
    surf(ax3, x1, x2, F, 'FaceAlpha', 0.8, 'EdgeColor', 'none');
    hold(ax3, 'on');

    % Add colored contours at the base
    nLevels = 40;
    levelList = linspace(minF, maxF, nLevels);
    colorMap = parula(nLevels);
    zBase = minF - 0.01 * (maxF - minF);

    xVec = x1(1, :);
    yVec = x2(:, 1)';

    for i = 1:nLevels
        level = levelList(i);
        C = contourc(xVec, yVec, F, [level level]);

        idx = 1;
        while idx < size(C, 2)
            nPoints = C(2, idx);
            if nPoints <= 0
                break;
            end

            xPoints = C(1, idx+1:idx+nPoints);
            yPoints = C(2, idx+1:idx+nPoints);

            plot3(ax3, xPoints, yPoints, zBase * ones(1, nPoints), ...
                  'LineWidth', 1, 'Color', colorMap(i, :));

            idx = idx + nPoints + 1;
        end
    end

    xlabel(ax3, 'x_1');
    ylabel(ax3, 'x_2');
    zlabel(ax3, 'f(x_1, x_2)');
    title(ax3, sprintf('PORTAL Combined (K=%d, Seed=%d)', PORTAL.K, PORTAL.Seed));
    colormap(ax3, parula);
    grid(ax3, 'on');
    view(ax3, -35, 30);
    box(ax3, 'on');
    axis(ax3, 'tight');
    pbaspect(ax3, [1 1 1]);
    square_axes_fill(ax3);
    rotate3d(ax3, 'on');

    fprintf('Plotting complete!\n');
end

%% ==================== Helper Functions ====================

function square_axes_fill(ax)
    % Make the axes' drawing region a centered square
    origUnits = ax.Units;
    ax.Units = 'normalized';
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    availW = outerpos(3) - ti(1) - ti(3);
    availH = outerpos(4) - ti(2) - ti(4);
    side = min(availW, availH);
    left = outerpos(1) + ti(1) + (availW - side) / 2;
    bottom = outerpos(2) + ti(2) + (availH - side) / 2;
    ax.Position = [left bottom side side];
    ax.Units = origUnits;
end

function square_axes_fill_with_colorbar(ax)
    % Modified version to maximize plot area while maintaining square aspect
    origUnits = ax.Units;
    ax.Units = 'normalized';

    margin = 0.08;
    colorbar_width = 0.03;
    colorbar_gap = 0.02;

    total_width = 1 - 2 * margin - colorbar_width - colorbar_gap;
    total_height = 1 - 2 * margin;

    side = min(total_width, total_height);

    left = margin + (total_width - side) / 2;
    bottom = margin + (total_height - side) / 2;

    ax.Position = [left bottom side side];
    ax.Units = origUnits;
end
