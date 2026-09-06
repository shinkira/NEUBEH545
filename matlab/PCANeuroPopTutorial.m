%% PCANeuroPopTutorial.m
%
% Principal Components Analysis of Neural Population Data
%
% This tutorial introduces PCA as a tool for understanding the geometry of
% neural population responses. It is designed to be run section by section
% (Ctrl+Enter / Cmd+Enter in each cell). Read the comments carefully -- the
% conceptual narrative and homework problems are embedded in the code.
%
% The tutorial builds in four stages:
%
%   Part I.   Why neural populations live in low-dimensional subspaces
%   Part II.  Covariance and its eigensystem -- the mathematical backbone of PCA
%   Part III. PCA on a simulated orientation-tuning population
%   Part IV.  PCA with two-dimensional stimuli (orientation × spatial frequency)
%   Part V.   Fixed-axis projection -- comparing two populations in a common space
%
% Prerequisites: linear algebra tutorial, stochastic processes tutorial
%
% Author:  Shin Kira (skira@fsu.edu)
% Created: 2026
%
% Dependencies: none (self-contained)

clear all
close all

%% ========================================================================
%  Part I.  Motivation: why neural data is high-dimensional but structured
%% ========================================================================

% Suppose you record simultaneously from N neurons while showing a visual
% stimulus. Each neuron's response at one moment is a single number. So
% the combined state of the population at that moment is a point in an
% N-dimensional space -- one axis per neuron.
%
% If N = 300, the population state lives in a 300-dimensional space. That
% sounds intractable. But neurons are not independent: they share inputs,
% are embedded in circuits, and respond to common stimulus features. These
% shared influences mean that the population does NOT wander freely through
% all 300 dimensions. Instead, its responses cluster around a
% low-dimensional surface -- a *manifold* -- embedded in that large space.
%
% PCA is a principled way to find the orientation of that manifold. It
% answers: what are the directions in neuron-space along which the
% population varies the most across stimuli?
%
% To build intuition, let's start with a 2-neuron "population" and
% visualize everything directly.

%% -------------------------------------------------------------------------
%  Example 1: two neurons, uncorrelated
%  -------------------------------------------------------------------------

rng(42)  % fix random seed for reproducibility

N_samples = 500;

% Two neurons with independent responses (different standard deviations)
neuron1 = randn(N_samples, 1) * 0.5;   % sd = 0.5
neuron2 = randn(N_samples, 1) * 2.0;   % sd = 2.0

Dist_indep = [neuron1, neuron2];

figure(1); clf
subplot(1,2,1)
plot(Dist_indep(:,1), Dist_indep(:,2), '.', 'MarkerSize', 4, 'Color', [0.4 0.4 0.8])
axis([-6 6 -6 6]); axis square
xlabel('Neuron 1 response'); ylabel('Neuron 2 response')
title('Independent neurons')

% The cloud of points is axis-aligned: knowing neuron 1''s response tells
% you nothing about neuron 2's response. The "natural" axes of this
% distribution are the x and y axes themselves.

%% -------------------------------------------------------------------------
%  Example 2: two neurons, correlated
%  -------------------------------------------------------------------------

% Now imagine both neurons receive a common input (e.g., a shared stimulus
% drive). We construct correlated responses by mixing independent sources.

common  = randn(N_samples, 1);
private = randn(N_samples, 1);

neuron1_corr = 0.8 * common + 0.2 * private;
neuron2_corr = 0.8 * common - 0.2 * private;

Dist_corr = [neuron1_corr, neuron2_corr];

subplot(1,2,2)
plot(Dist_corr(:,1), Dist_corr(:,2), '.', 'MarkerSize', 4, 'Color', [0.8 0.4 0.4])
axis([-3 3 -3 3]); axis square
xlabel('Neuron 1 response'); ylabel('Neuron 2 response')
title('Correlated neurons (shared input)')

% The correlated cloud is elongated along the diagonal. The "natural" axes
% are no longer the x and y axes -- they're rotated. PCA finds these
% natural axes.

% Homework question 1.
% (a) What does the orientation of the elongated cloud tell you about the
%     relationship between the two neurons?
% (b) If common input is much stronger than private noise, what shape does
%     the cloud approach? What does PCA return in that limit?
% (c) What would the cloud look like if neuron2 = -neuron1 + noise?
%     How would PCA change?

%% ========================================================================
%  Part II.  The covariance matrix and its eigensystem
%% ========================================================================

% The covariance matrix captures pairwise relationships between all pairs
% of neurons. For a population recorded over many stimuli, the (i,j) entry
% of the covariance matrix is:
%
%   Cov(i,j) = <r_i * r_j> - <r_i><r_j>
%
% where <.> denotes the average across stimuli (or trials). The diagonal
% entries are the variances of each neuron; off-diagonal entries reflect
% how much two neurons co-vary. Let's compute the covariance matrix for
% our two example distributions.

C_indep = cov(Dist_indep);
C_corr  = cov(Dist_corr);

fprintf('\nCovariance matrix, independent neurons:\n');
disp(C_indep)
fprintf('Covariance matrix, correlated neurons:\n');
disp(C_corr)

% For the independent case, the off-diagonal entries are near zero (small
% sampling noise only). For the correlated case, they are large and
% positive, reflecting the shared drive.

% The key insight: the *eigenvectors* of the covariance matrix point along
% the natural axes of the data cloud. The corresponding *eigenvalues* tell
% you how much variance falls along each eigenvector.

[V_indep, D_indep] = eig(C_indep);
[V_corr,  D_corr]  = eig(C_corr);

% Sort by descending eigenvalue (MATLAB's eig doesn't guarantee order)
[d_indep, idx] = sort(diag(D_indep), 'descend');
V_indep = V_indep(:, idx);

[d_corr, idx] = sort(diag(D_corr), 'descend');
V_corr = V_corr(:, idx);

fprintf('Eigenvalues (independent): %.3f, %.3f\n', d_indep(1), d_indep(2));
fprintf('Eigenvalues (correlated):  %.3f, %.3f\n', d_corr(1),  d_corr(2));

% Now superimpose the eigenvectors on the scatter plots
figure(1)
subplot(1,2,1); hold on
scale = 2;
quiver(0, 0, scale*V_indep(1,1)*sqrt(d_indep(1)), scale*V_indep(2,1)*sqrt(d_indep(1)), ...
    0, 'r', 'LineWidth', 2.5, 'MaxHeadSize', 0.5)
quiver(0, 0, scale*V_indep(1,2)*sqrt(d_indep(2)), scale*V_indep(2,2)*sqrt(d_indep(2)), ...
    0, 'k', 'LineWidth', 2.5, 'MaxHeadSize', 0.5)
hold off

subplot(1,2,2); hold on
quiver(0, 0, scale*V_corr(1,1)*sqrt(d_corr(1)), scale*V_corr(2,1)*sqrt(d_corr(1)), ...
    0, 'r', 'LineWidth', 2.5, 'MaxHeadSize', 0.5)
quiver(0, 0, scale*V_corr(1,2)*sqrt(d_corr(2)), scale*V_corr(2,2)*sqrt(d_corr(2)), ...
    0, 'k', 'LineWidth', 2.5, 'MaxHeadSize', 0.5)
legend('Data', 'PC1', 'PC2', 'Location', 'northwest')
hold off

% For the independent case the eigenvectors line up with the coordinate
% axes (as expected). For the correlated case they are rotated ~45 degrees,
% aligned with the main axis of the elongated cloud.

% Homework question 2.
% (a) The length of each arrow is scaled by sqrt(eigenvalue). Why is that
%     a meaningful choice? What does it represent geometrically?
% (b) The total variance of the population (summed across neurons) equals
%     the sum of all eigenvalues. Verify this for both distributions using
%     the diagonal of the covariance matrix and the eigenvalues.
% (c) PCA is equivalent to finding a rotation of the coordinate axes such
%     that the covariance matrix becomes diagonal in the new frame. Convince
%     yourself of this by computing V' * C * V for the correlated case.
%     What do you find?

%% ========================================================================
%  Part III.  PCA on a simulated orientation-tuning population
%% ========================================================================

% Now let's move to a realistic neural setting: a population of N neurons
% tuned to stimulus orientation. This mirrors a cortical recording in which
% many V1 neurons respond to oriented gratings with different preferred
% orientations.

%% -------------------------------------------------------------------------
%  Simulate population responses
%  -------------------------------------------------------------------------

N         = 300;           % number of neurons
baseline  = 0.1;           % baseline response (e.g., dF/F or normalized spike rate)
gain      = 1.0;           % response gain

% Preferred orientations, uniformly tiling 0..180 degrees
% (orientation is a 180-degree variable because a grating at 0 deg looks
% the same as one at 180 deg)
pref_oris = linspace(0, 179.9, N)';   % N x 1, degrees

% Stimulus orientations to present
stim_oris = 0:1:179;                  % 180 orientations, 1-deg steps

% Orientation tuning: von Mises-like function, 180-deg periodic.
% The response of neuron n to stimulus s is:
%   R(n,s) = baseline + gain * exp( kappa * cos( 2*(theta_s - theta_n) ) )
%
% kappa controls tuning width: larger kappa = sharper tuning.
kappa = 3 * ones(N, 1);

deg2rad    = pi / 180;
theta_pref = pref_oris * deg2rad;     % N x 1
theta_stim = stim_oris * deg2rad;     % 1 x S

S = numel(stim_oris);
R = zeros(N, S);

for s = 1:S
    dtheta  = theta_stim(s) - theta_pref;      % N x 1
    R(:, s) = baseline + gain * exp(kappa .* cos(2 * dtheta));
end

% Normalize each neuron's responses so that its maximum = baseline + gain.
% This removes differences in absolute firing rate across neurons and lets
% us focus on the shape of the population code.
R = R ./ max(R, [], 2) * (baseline + gain);

fprintf('Population response matrix R: %d neurons x %d stimuli\n', size(R,1), size(R,2));

%% -------------------------------------------------------------------------
%  Visualize the population response matrix
%  -------------------------------------------------------------------------

figure(2); clf
set(gcf, 'OuterPosition', [50 100 700 600])
imagesc(stim_oris, 1:N, R);
colormap(parula); colorbar;
xlabel('Stimulus orientation (deg)');
ylabel('Neuron index');
title('Population response matrix R  (neurons × stimuli)', 'FontSize', 16);

% Each row is the tuning curve of one neuron. You can see the diagonal
% band of high activity: each neuron responds strongly when the stimulus
% orientation matches its preference.

%% -------------------------------------------------------------------------
%  Example tuning curves
%  -------------------------------------------------------------------------

example_neurons = [1 51 101 151 201 251];   % neurons with different preferences
colors = lines(numel(example_neurons));

figure(3); clf
set(gcf, 'OuterPosition', [50 100 750 500])
hold on
for ii = 1:numel(example_neurons)
    n = example_neurons(ii);
    plot(stim_oris, R(n,:), 'LineWidth', 2, 'Color', colors(ii,:));
end
hold off
xlabel('Stimulus orientation (deg)');
ylabel('Normalized response');
title('Example tuning curves', 'FontSize', 16);
legend(arrayfun(@(n) sprintf('Neuron %d (pref %.0f°)', n, pref_oris(n)), ...
    example_neurons, 'UniformOutput', false), 'Location', 'northeast');
xlim([0 180]);

%% -------------------------------------------------------------------------
%  Run PCA
%  -------------------------------------------------------------------------

% Convention: rows of X are observations (stimuli), columns are features (neurons).
% We mean-center across stimuli before running PCA -- this removes any
% constant offset that is the same for every stimulus.

X  = R.';             % [S x N] -- each row is one stimulus condition
Xc = X - mean(X, 1); % subtract per-neuron mean across stimuli

[coeff, score, latent] = pca(Xc);
% coeff:  [N x N] -- columns are principal component directions (in neuron space)
% score:  [S x N] -- projections of each stimulus onto each PC
% latent: [N x 1] -- eigenvalues (variance explained by each PC)

explVar    = 100 * latent / sum(latent);
cumExplVar = cumsum(explVar);

%% -------------------------------------------------------------------------
%  Scree plot -- how many dimensions does the representation need?
%  -------------------------------------------------------------------------

figure(4); clf
set(gcf, 'OuterPosition', [800 100 700 550])

subplot(2,1,1)
plot(1:numel(explVar), cumExplVar, '-o', 'LineWidth', 1.5, ...
    'MarkerSize', 4, 'Color', [0.2 0.5 0.8])
yline(90, '--k', '90%', 'LabelHorizontalAlignment', 'left');
yline(99, '--r', '99%', 'LabelHorizontalAlignment', 'left');
xlabel('Number of PCs');
ylabel('Cumulative variance (%)');
title('Explained variance (orientation-only population)', 'FontSize', 14);
grid on; ylim([0 101]);

subplot(2,1,2)
plot(1:20, explVar(1:20), '-o', 'LineWidth', 1.5, ...
    'MarkerSize', 5, 'Color', [0.8 0.3 0.2])
xlabel('PC index');
ylabel('Variance explained (%)');
title('Per-PC variance (first 20 PCs)', 'FontSize', 14);
grid on;

fprintf('\nTop 3 PCs explain: %.1f%%, %.1f%%, %.1f%%\n', ...
    explVar(1), explVar(2), explVar(3));

% If you ran the simulation correctly, you should find that the top 2--3
% PCs explain almost all the variance. Why? Because all 300 neurons are
% tuned to orientation in the same smooth way -- the population only needs
% a small number of "basis functions" to represent all 180 orientations.

% Homework question 3.
% (a) How many PCs are needed to explain 99% of the variance? Compare this
%     to the number of neurons (N = 300) and the number of stimuli (S = 180).
%     What does this tell you about the intrinsic dimensionality of the
%     orientation representation?
% (b) Repeat the simulation with kappa = 1 (broader tuning) and kappa = 8
%     (sharper tuning). How does the explained variance plot change? Why?
% (c) Add Gaussian noise to R (noise_sigma = 0.1 * randn(size(R))) before
%     running PCA. How does the scree plot change? What does this tell you
%     about what PCA "sees" in noisy data?

%% -------------------------------------------------------------------------
%  The orientation ring: PC1-PC2 space
%  -------------------------------------------------------------------------

% Color each stimulus by its orientation using a cyclic (HSV) colormap.
% Since orientation is periodic (0 deg == 180 deg visually), a cyclic
% colormap is the right choice.

angles_color = linspace(0, 2*pi, S+1);
angles_color(end) = [];
cmap_ori = hsv2rgb([angles_color'/(2*pi), ones(S,1), ones(S,1)]);

figure(5); clf
set(gcf, 'OuterPosition', [50 100 700 700])
hold on

% Draw the connecting line (stimuli in order) to show the ring structure
plot(score(:,1), score(:,2), '-k', 'LineWidth', 1.0)
plot([score(end,1) score(1,1)], [score(end,2) score(1,2)], '-k', 'LineWidth', 1.0)

for s = 1:S
    scatter(score(s,1), score(s,2), 40, cmap_ori(s,:), 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.3);
end

xlabel(sprintf('PC1 (%.1f%%)', explVar(1)));
ylabel(sprintf('PC2 (%.1f%%)', explVar(2)));
title('Population response geometry: PC1 vs PC2', 'FontSize', 16);

colormap(gca, cmap_ori);
c = colorbar;
tick_oris = 0:30:180;
c.Ticks = tick_oris / 180;
c.TickLabels = string(tick_oris) + "°";
c.Label.String = 'Stimulus orientation';

axis equal; grid on; hold off

% The population response traces out a *ring* (approximately a circle) in
% PC1-PC2 space. This is the "orientation manifold" of V1-like populations.
%
% Why a ring? Orientation is a periodic variable. As the stimulus rotates
% from 0 to 180 degrees, the population activity vector rotates through
% neuron space -- and when mapped to PC space it sweeps out a closed loop.
% The precise shape is approximately circular because of the smooth,
% periodic nature of von Mises tuning.

%% -------------------------------------------------------------------------
%  What do PC1 and PC2 look like in neuron space?
%  -------------------------------------------------------------------------

% The principal components are vectors in neuron space. Each entry
% (coeff(n, k)) tells you how much neuron n contributes to PC k.
% Let's plot the first two PCs as functions of the neurons' preferred
% orientations -- this reveals their interpretation.

figure(6); clf
set(gcf, 'OuterPosition', [800 100 750 500])
hold on
plot(pref_oris, coeff(:,1), 'b-', 'LineWidth', 2);
plot(pref_oris, coeff(:,2), 'r-', 'LineWidth', 2);
xlabel('Preferred orientation of neuron (deg)');
ylabel('PC loading');
title('PC loadings as a function of preferred orientation', 'FontSize', 16);
legend('PC1', 'PC2');
xlim([0 180]); grid on; hold off

% PC1 and PC2 look like sinusoids -- cosine and sine -- as functions of
% preferred orientation. This makes geometric sense: they are the first two
% Fourier components of the orientation "circle". Together they encode
% which direction on the ring the population is pointing.
%
% This gives us a beautiful interpretation: PC1 and PC2 form a 2D
% coordinate system that captures orientation as an angle. Reading out
% the stimulus orientation from the population is equivalent to computing
% atan2(score2, score1).

% Verify: can we recover the stimulus from PC coordinates alone?
est_ori_rad = atan2(score(:,2), score(:,1));   % recover angle from PC plane
est_ori_deg = mod(est_ori_rad * 180/pi, 360);
% atan2 returns values in (-180, 180), and our orientation is periodic over
% 180 deg (not 360), so we need to fold back:
est_ori_deg = mod(est_ori_deg, 180);

figure(7); clf
set(gcf, 'OuterPosition', [50 100 650 500])
plot(stim_oris, est_ori_deg, '.', 'MarkerSize', 5, 'Color', [0.3 0.6 0.9])
hold on; plot([0 180], [0 180], '-k', 'LineWidth', 1.5); hold off
xlabel('True stimulus orientation (deg)');
ylabel('Estimated orientation from PCs (deg)');
title('Orientation decoded from PC1-PC2 alone', 'FontSize', 16);
grid on; axis square

% Homework question 4.
% (a) Why does the decoded orientation have a discontinuity or "fold" at
%     some stimulus values? What is the relationship between the 180-deg
%     periodicity of orientation and the 360-deg range of atan2?
% (b) How would you modify the decoding to handle the 180-deg vs 360-deg
%     ambiguity perfectly? Hint: think about how a 2*theta wrapping affects
%     the response.
% (c) If you added a third neuron class tuned to colors (not orientation),
%     where in the PC space would color responses appear? What would the
%     scree plot look like?

%% -------------------------------------------------------------------------
%  Effect of tuning width on population geometry
%  -------------------------------------------------------------------------

% Let's compare populations with different kappa values side by side.

figure(8); clf
set(gcf, 'OuterPosition', [50 100 1200 450])

kappa_vals  = [1, 3, 8];
panel_titles = {'Broad tuning (κ = 1)', 'Medium tuning (κ = 3)', 'Sharp tuning (κ = 8)'};

for ki = 1:3
    kv = kappa_vals(ki);
    kappa_v = kv * ones(N, 1);

    Rv = zeros(N, S);
    for s = 1:S
        dtheta   = theta_stim(s) - theta_pref;
        Rv(:, s) = baseline + gain * exp(kappa_v .* cos(2 * dtheta));
    end
    Rv = Rv ./ max(Rv, [], 2) * (baseline + gain);

    Xv  = Rv.';
    Xvc = Xv - mean(Xv, 1);
    [~, sv, lv] = pca(Xvc);

    ev = 100 * lv / sum(lv);

    subplot(1,3,ki)
    plot(sv(:,1), sv(:,2), '-k', 'LineWidth', 0.8); hold on
    plot([sv(end,1) sv(1,1)], [sv(end,2) sv(1,2)], '-k', 'LineWidth', 0.8);
    for s = 1:S
        scatter(sv(s,1), sv(s,2), 20, cmap_ori(s,:), 'filled');
    end
    hold off
    axis equal; grid on;
    xlabel('PC1'); ylabel('PC2');
    title(sprintf('%s\n%.1f%% + %.1f%%', panel_titles{ki}, ev(1), ev(2)), 'FontSize', 13);
end

% Notice that all three populations form a ring, but the *radius* and
% how variance is distributed across PCs differs. With sharper tuning,
% each neuron responds to fewer stimuli, so the responses are more
% "spiky" and higher PCs carry more variance (the ring is rougher). With
% broader tuning, the responses are smoother and nearly all variance
% collapses into the first two PCs.

%% ========================================================================
%  Part IV.  PCA with two-dimensional stimuli: orientation × spatial frequency
%% ========================================================================

% Real visual neurons are tuned to both orientation AND spatial frequency
% (SF). A neuron preferring vertical gratings at 2 c/deg responds poorly
% to horizontal gratings or to very low/high spatial frequencies. When we
% vary both dimensions independently, the population response traces out a
% more complex surface in neuron space -- a torus.

%% -------------------------------------------------------------------------
%  Build an orientation × SF population
%  -------------------------------------------------------------------------

N_ori = 80;    % preferred orientations
N_sf  = 10;    % preferred spatial frequencies per orientation
N2    = N_ori * N_sf;

pref_oris2_base = linspace(0, 179.9, N_ori)';
sf_grid_base    = logspace(-2, 1, N_sf);   % 0.01 to 10 c/deg

[ori_idx2, sf_idx2] = ndgrid(1:N_ori, 1:N_sf);
pref_oris2 = pref_oris2_base(ori_idx2(:));   % N2 x 1
pref_sfs2  = sf_grid_base(sf_idx2(:))';      % N2 x 1

% Stimulus grid: all combinations of orientation and SF
stim_oris2 = 0:15:165;    % 12 orientations
stim_sfs2  = logspace(-2, 1, 8);   % 8 SFs

[ori_sg, sf_sg] = ndgrid(1:numel(stim_oris2), 1:numel(stim_sfs2));
stim_ori_list = stim_oris2(ori_sg(:));   % n_cond x 1
stim_sf_list  = stim_sfs2(sf_sg(:));     % n_cond x 1
n_cond2       = numel(stim_ori_list);

% Tuning parameters
kappa2    = 3 * ones(N2, 1);   % orientation selectivity
sf_sigma2 = 0.5 * ones(N2, 1); % SF bandwidth in log10 units

deg2rad2     = pi/180;
theta_pref2  = pref_oris2 * deg2rad2;
log_pref_sfs = log10(pref_sfs2);

% Compute population responses
R2 = zeros(N2, n_cond2);

for c = 1:n_cond2
    theta_stim2  = stim_ori_list(c) * deg2rad2;
    dtheta2      = theta_stim2 - theta_pref2;                  % N2 x 1
    ori_tuning2  = exp(kappa2 .* cos(2 * dtheta2));

    log_sf_stim = log10(stim_sf_list(c));
    dlog_sf      = log_sf_stim - log_pref_sfs;                 % N2 x 1
    sf_tuning2   = exp(-(dlog_sf.^2) ./ (2 * sf_sigma2.^2));

    R2(:, c) = baseline + gain * (ori_tuning2 .* sf_tuning2);
end

fprintf('Orientation × SF population: %d neurons, %d stimulus conditions\n', N2, n_cond2);

%% -------------------------------------------------------------------------
%  PCA on the orientation × SF population
%  -------------------------------------------------------------------------

X2  = R2.';
X2c = X2 - mean(X2, 1);

[~, score2, latent2] = pca(X2c);
explVar2 = 100 * latent2 / sum(latent2);

fprintf('Top 5 PCs explain: %.1f%%, %.1f%%, %.1f%%, %.1f%%, %.1f%%\n', ...
    explVar2(1), explVar2(2), explVar2(3), explVar2(4), explVar2(5));

%% -------------------------------------------------------------------------
%  Color coding: orientation = hue, SF = saturation
%  -------------------------------------------------------------------------

nOri_c2 = numel(stim_oris2);
hues2    = linspace(0, 1, nOri_c2+1)';
hues2(end) = [];

[~, ori_idx_for_cond] = ismember(stim_ori_list, stim_oris2);

sf_min2   = min(stim_sf_list);
sf_max2   = max(stim_sf_list);
sf_norm2  = (stim_sf_list - sf_min2) / (sf_max2 - sf_min2);
saturation = 0.2 + 0.8 * sf_norm2;   % low SF -> pale, high SF -> vivid

colors2 = zeros(n_cond2, 3);
for i = 1:n_cond2
    colors2(i,:) = hsv2rgb([hues2(ori_idx_for_cond(i)), saturation(i), 1]);
end

%% -------------------------------------------------------------------------
%  3D PC plot: orientation × SF manifold
%  -------------------------------------------------------------------------

figure(9); clf
set(gcf, 'OuterPosition', [50 100 750 750])
hold on

plot3(score2(:,1), score2(:,2), score2(:,3), '-k', 'LineWidth', 0.8);

for i = 1:n_cond2
    scatter3(score2(i,1), score2(i,2), score2(i,3), ...
        70, colors2(i,:), 'filled', 'MarkerEdgeColor','k', 'LineWidth', 0.3);
end

xlabel(sprintf('PC1 (%.1f%%)', explVar2(1)));
ylabel(sprintf('PC2 (%.1f%%)', explVar2(2)));
zlabel(sprintf('PC3 (%.1f%%)', explVar2(3)));
title('Orientation × SF population: PC1-PC2-PC3 manifold', 'FontSize', 16);

cmap2 = hsv2rgb([hues2, ones(nOri_c2,1), ones(nOri_c2,1)]);
colormap(gca, cmap2);
c2 = colorbar;
tick_oris2    = 0:30:165;
c2.Ticks      = tick_oris2 / max(stim_oris2);
c2.TickLabels = string(tick_oris2) + "°";
c2.Label.String = 'Orientation (hue), SF (saturation)';

grid on; axis equal; view(3); hold off

% With two stimulus dimensions, the manifold is no longer a simple ring.
% Instead it takes the form of a torus: orientation wraps around one
% direction (the ring we saw before), and SF varies along the "tube"
% of the torus. The dimensionality of the representation has increased --
% we now need more PCs to capture the full variance.

%% -------------------------------------------------------------------------
%  Scree plot comparison: 1D vs 2D stimuli
%  -------------------------------------------------------------------------

figure(10); clf
set(gcf, 'OuterPosition', [800 100 700 500])
hold on
plot(1:20, cumExplVar(1:20), '-o', 'LineWidth', 1.8, ...
    'MarkerSize', 5, 'Color', [0.2 0.5 0.9], 'DisplayName', 'Orientation only');
plot(1:20, cumsum(explVar2(1:20)), '-s', 'LineWidth', 1.8, ...
    'MarkerSize', 5, 'Color', [0.9 0.3 0.2], 'DisplayName', 'Orientation × SF');
yline(90, '--k');
hold off
xlabel('Number of PCs');
ylabel('Cumulative variance (%)');
title('Dimensionality: 1D vs 2D stimuli', 'FontSize', 16);
legend('Location', 'southeast'); grid on;

% Homework question 5.
% (a) Why does the 2D stimulus set require more PCs to reach 90% explained
%     variance than the 1D set? Relate your answer to the intrinsic
%     dimensionality of the stimulus space.
% (b) If you were to record from a real V1 population using random stimuli
%     (pink noise images), would you expect the scree plot to decay faster
%     or slower than the orientation-only case? Why?
% (c) How would the torus shape change if all neurons had the same preferred
%     SF (i.e., N_sf = 1)? What would the scree plot look like?

%% ========================================================================
%  Part V.  Fixed-axis projection: comparing two populations
%% ========================================================================

% A powerful use of PCA is to compare how a population responds to two
% different conditions. If we run PCA separately on each condition, the
% PC axes may differ between conditions, making direct comparison hard.
%
% A better approach: define the PC axes using one reference (baseline)
% condition, then project BOTH populations onto those same axes. Changes
% in the shape or location of the resulting clouds then directly reflect
% true changes in population geometry, not just rotation of the PC frame.

%% -------------------------------------------------------------------------
%  Simulate two conditions: baseline and "biased" (cardinal SF gain)
%  -------------------------------------------------------------------------

% In the "biased" condition, neurons preferring cardinal orientations
% (0 deg and 90 deg) respond more strongly to their preferred SF than
% neurons preferring oblique orientations. This is inspired by psychophysical
% and physiological evidence for cardinal-oblique anisotropies in V1.

sf_gain_amp = 0.5;   % strength of the cardinal bias

% Cardinal modulation: neurons tuned near 0 or 90 deg get a gain boost.
% cos(2*theta)^2 peaks at 0 and 90 deg (the cardinals) and is zero at
% the obliques (45 and 135 deg).
cardinal_mod = 1 + sf_gain_amp * (cos(2 * theta_pref2)).^2;   % N2 x 1

% Baseline population (no cardinal bias)
R_base = R2;   % already computed above

% Biased population: SF tuning is amplified for cardinals
R_bias = zeros(N2, n_cond2);
for c = 1:n_cond2
    theta_stim2  = stim_ori_list(c) * deg2rad2;
    dtheta2      = theta_stim2 - theta_pref2;
    ori_tuning2  = exp(kappa2 .* cos(2 * dtheta2));

    log_sf_stim = log10(stim_sf_list(c));
    dlog_sf      = log_sf_stim - log_pref_sfs;
    sf_tuning2   = exp(-(dlog_sf.^2) ./ (2 * sf_sigma2.^2));

    % Apply cardinal-dependent SF gain
    sf_tuning_biased = cardinal_mod .* sf_tuning2;

    R_bias(:, c) = baseline + gain * (ori_tuning2 .* sf_tuning_biased);
end

%% -------------------------------------------------------------------------
%  PCA on baseline, project biased onto the same axes
%  -------------------------------------------------------------------------

% Step 1: compute the baseline PC axes
X_base  = R_base.';
X_basec = X_base - mean(X_base, 1);   % center using baseline mean

[coeff_base, score_base, latent_base] = pca(X_basec);
explVar_base = 100 * latent_base / sum(latent_base);

% Step 2: project the biased population using the SAME mean and axes
X_bias  = R_bias.';
X_biasc = X_bias - mean(X_base, 1);   % subtract BASELINE mean (not biased mean)
score_bias_fixed = X_biasc * coeff_base;   % project onto baseline PCs

% By subtracting the baseline mean and projecting onto baseline PCs,
% any difference in score_bias vs score_base directly reflects changes
% in population activity patterns, not an artifact of re-centering or
% re-rotating PCs.

%% -------------------------------------------------------------------------
%  Compare the two manifolds in the shared PC space
%  -------------------------------------------------------------------------

figure(11); clf
set(gcf, 'OuterPosition', [50 100 1350 650])

subplot(1,2,1)
hold on
plot3(score_base(:,1), score_base(:,2), score_base(:,3), '-k', 'LineWidth', 0.8)
for i = 1:n_cond2
    scatter3(score_base(i,1), score_base(i,2), score_base(i,3), ...
        60, colors2(i,:), 'filled', 'MarkerEdgeColor','k', 'LineWidth', 0.3);
end
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title(sprintf('Baseline population\n(%.1f/%.1f/%.1f%%)', ...
    explVar_base(1), explVar_base(2), explVar_base(3)), 'FontSize', 14);
grid on; axis equal; view(3); hold off
colormap(gca, cmap2);

subplot(1,2,2)
hold on
plot3(score_bias_fixed(:,1), score_bias_fixed(:,2), score_bias_fixed(:,3), ...
    '-k', 'LineWidth', 0.8)
for i = 1:n_cond2
    scatter3(score_bias_fixed(i,1), score_bias_fixed(i,2), score_bias_fixed(i,3), ...
        60, colors2(i,:), 'filled', 'MarkerEdgeColor','k', 'LineWidth', 0.3);
end
xlabel('PC1 (baseline)'); ylabel('PC2 (baseline)'); zlabel('PC3 (baseline)');
title(sprintf('Biased population (cardinal SF gain = %.1f)\nprojected onto baseline PCs', ...
    sf_gain_amp), 'FontSize', 14);
grid on; axis equal; view(3); hold off
colormap(gca, cmap2);

sgtitle('Fixed-axis comparison: baseline vs cardinal-biased population', 'FontSize', 16);

% Comparing the two panels, you should notice that the biased population's
% manifold has a different shape than the baseline in the shared PC space.
% Cardinal orientations (red/blue points) are displaced, reflecting the
% extra SF gain their neurons receive. Oblique orientations are relatively
% unchanged.
%
% This is the key advantage of the fixed-axis approach: the deformation of
% the manifold is directly interpretable as a change in the population
% code, not a difference in how PCA chose to orient its axes.

%% -------------------------------------------------------------------------
%  Quantify the deformation: distance from baseline in PC space
%  -------------------------------------------------------------------------

% For each stimulus condition, compute the Euclidean distance (in the
% top-3-PC subspace) between where the baseline and biased populations land.
dist_pc3 = sqrt(sum((score_base(:,1:3) - score_bias_fixed(:,1:3)).^2, 2));

% Sort stimulus conditions by orientation for plotting
[sorted_oris, sort_idx] = sort(stim_ori_list);

figure(12); clf
set(gcf, 'OuterPosition', [800 100 700 500])
hold on
scatter(sorted_oris, dist_pc3(sort_idx), 60, colors2(sort_idx,:), ...
    'filled', 'MarkerEdgeColor','k', 'LineWidth', 0.3);
hold off
xlabel('Stimulus orientation (deg)');
ylabel('Distance in PC1-PC3 space (baseline vs biased)');
title('Manifold deformation: where does the cardinal bias show up?', 'FontSize', 14);
xlim([-5 170]); grid on;

% The deformation should be largest for cardinal orientations (0 and 90 deg)
% and smallest for obliques (45 deg). This quantitative readout gives you
% a geometric signature of the population-level effect of the cardinal bias.

% Homework question 6.
% (a) Vary sf_gain_amp from 0 to 2. How does the shape of figure 11 change?
%     Is there a threshold below which the bias is undetectable in the
%     PC plot?
% (b) The fixed-axis projection subtracts the *baseline* mean before
%     projecting the biased data. What would happen if you instead
%     subtracted the biased population's own mean? Would the comparison
%     still be valid? When might you prefer one over the other?
% (c) Design an experiment using synthetic data: generate a biased
%     population with a bias that affects only low-SF responses (not high
%     SF). Predict how figures 11 and 12 would look, then verify with code.

%% ========================================================================
%  Summary
%% ========================================================================
%
%  In this tutorial we have covered:
%
%  1. Neural population responses form low-dimensional manifolds in the
%     high-dimensional space of all possible population states.
%
%  2. The covariance matrix captures pairwise correlations between neurons.
%     Its eigenvectors define "natural" axes (PCs); eigenvalues give the
%     variance along each axis.
%
%  3. For an orientation-tuning population, PCA recovers a *ring* in
%     PC1-PC2 space. The ring reflects the 180-deg periodicity of the
%     stimulus variable. PC1 and PC2 have cosine/sine loadings over
%     preferred orientation and can be used to decode stimulus orientation
%     directly.
%
%  4. Adding a second stimulus dimension (spatial frequency) expands the
%     manifold from a ring to a torus, requiring more PCs to capture the
%     full variance.
%
%  5. Fixed-axis projection -- defining PC axes from a reference condition
%     and projecting a second condition onto those axes -- allows direct,
%     interpretable comparison of population geometry across conditions.
%
%  Further reading:
%   - Cunningham & Yu (2014). Dimensionality reduction for large-scale
%     neural recordings. Nature Neuroscience.
%   - Jazayeri & Ostojic (2021). Interpreting neural computations by
%     examining intrinsic and embedding dimensionality. Current Opinion
%     in Neurobiology.
%   - Rieke et al., PCATutorial.m (this course).
