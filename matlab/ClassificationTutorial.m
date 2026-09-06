%% ClassificationTutorial.m
%
% Pattern Discrimination and Classification of Neural Population Data
%
% This tutorial asks a question that PCA cannot answer. PCA finds the
% directions in neuron-space along which responses vary the most. But
% variance is not the same thing as usefulness. If you want to know which
% of two stimuli was presented on a given trial, you need directions that
% *separate* the two response distributions -- and the most separating
% direction is often not the highest-variance one.
%
% We build up from a single neuron to a full population, and from a simple
% threshold rule to regularized, cross-validated multivariate classifiers.
%
%   Part I.    Signal detection theory: one neuron, ROC, AUC, and d-prime
%   Part II.   Neurometric functions: which neurons carry the information?
%   Part III.  Linear discriminant analysis: from one neuron to many
%   Part IV.   Logistic regression (a GLM classifier)
%   Part V.    Support vector machines and the margin
%   Part VI.   Cross-validation, overfitting, and regularization
%   Part VII.  Decoding geometry: signal vs. noise correlations
%   Part VIII. Nonlinear classifiers and a head-to-head comparison
%
% Throughout, we use the same simulated visual population as the PCA
% tutorial: neurons with von Mises orientation tuning, now with realistic
% trial-to-trial variability. The task is FINE ORIENTATION DISCRIMINATION:
% on each trial the animal sees a grating at either 90 - dtheta/2 or
% 90 + dtheta/2 degrees, and must report which.
%
% Run this section by section (Ctrl+Enter / Cmd+Enter). The conceptual
% narrative and the homework problems are embedded in the comments.
%
% Prerequisites: linear algebra tutorial, PCA tutorial, stochastic
%                processes tutorial
%
% Author:  Shin Kira (skira@fsu.edu)
% Created: 2026
%
% Dependencies: core MATLAB only, with one exception. Part V uses fitcsvm
%               (Statistics and Machine Learning Toolbox); the script
%               detects whether it is available and falls back to a
%               hand-written margin classifier if not. Everything else --
%               ROC, ranks, LDA, IRLS, kNN, cross-validation -- is
%               implemented from scratch in the helper functions at the
%               end of this file, so you can read exactly what each
%               method does.

clear all
close all

rng(7)   % fix the random seed so that every run reproduces the figures


%% ========================================================================
%  Part 0.  The simulated population and the discrimination task
%% ========================================================================

% We keep the encoding model deliberately simple, because the point of this
% tutorial is the DECODER, not the encoder. Each neuron has a von Mises
% orientation tuning curve that is periodic with 180 degrees:
%
%   mu_i(theta) = baseline + gain * exp(kappa * cos(2*(theta - phi_i))) / exp(kappa)
%
% where phi_i is neuron i's preferred orientation and kappa sets the tuning
% width. The division by exp(kappa) normalizes the peak to 1 so that gain
% is interpretable.
%
% What is new here, and what makes classification meaningful, is NOISE.
% Real neurons do not produce the same response twice. We use the standard
% cortical approximation that variance grows in proportion to the mean:
%
%   var_i = fano * mu_i
%
% A Fano factor of 1 corresponds to Poisson spiking. Values slightly above
% 1 are typical in visual cortex.

N        = 80;                           % number of neurons
kappa    = 3;                            % orientation tuning concentration
baseline = 2;                            % baseline rate (spikes/s)
gain     = 10;                           % peak evoked rate above baseline
fano     = 1.5;                          % variance-to-mean ratio

pref_oris = linspace(0, 180, N+1)';      % preferred orientations, 0..180
pref_oris = pref_oris(1:end-1);          % drop the duplicate endpoint

% ----- The discrimination task -----
% These values were chosen so the problem sits in the interesting regime:
% hard enough that no single neuron solves it, easy enough that the
% population does not saturate at 100 percent. If you make the task too
% easy, every method below scores 1.0 and the comparisons become vacuous --
% a failure mode worth remembering when you design your own simulations.
theta_center = 90;                       % reference orientation (deg)
dtheta       = 4;                        % angular separation (deg)

ori_A = theta_center - dtheta/2;         % 88 deg   -> class y = 0
ori_B = theta_center + dtheta/2;         % 92 deg   -> class y = 1

n_trials = 400;                          % trials per stimulus

% Mean population response to each of the two stimuli
mu_A = tuning_mean(pref_oris, kappa, ori_A, baseline, gain);
mu_B = tuning_mean(pref_oris, kappa, ori_B, baseline, gain);

% For Parts I-VI we start with INDEPENDENT noise across neurons. Part VII
% relaxes this, and you will see that the assumption matters enormously.
L_indep = eye(N);

X_A = sample_trials(mu_A, n_trials, fano, L_indep);   % n_trials x N
X_B = sample_trials(mu_B, n_trials, fano, L_indep);

% Stack into the standard supervised-learning format: rows are trials,
% columns are neurons (features), y is the class label.
X = [X_A; X_B];
y = [zeros(n_trials,1); ones(n_trials,1)];

fprintf('Design matrix X is %d trials x %d neurons\n', size(X,1), size(X,2));
fprintf('Task: %g deg vs %g deg (separation %g deg)\n', ori_A, ori_B, dtheta);

% ----- Look at the two mean population patterns -----
figure(1); clf
subplot(2,1,1); hold on
plot(pref_oris, mu_A, 'LineWidth', 2, 'Color', [0.20 0.40 0.85])
plot(pref_oris, mu_B, 'LineWidth', 2, 'Color', [0.85 0.30 0.25])
xlabel('Preferred orientation (deg)'); ylabel('Mean rate (spikes/s)')
legend(sprintf('%g deg', ori_A), sprintf('%g deg', ori_B), 'Location','northeast')
title('Mean population response to the two stimuli')
xlim([0 180]); set(gca,'XTick',0:30:180)

subplot(2,1,2); hold on
plot(pref_oris, mu_B - mu_A, 'k', 'LineWidth', 2)
plot([0 180], [0 0], ':', 'Color', [0.5 0.5 0.5])
xlabel('Preferred orientation (deg)'); ylabel('\Delta mean rate')
title('Difference between the two mean patterns -- the SIGNAL')
xlim([0 180]); set(gca,'XTick',0:30:180)

% Study the bottom panel carefully. The two mean patterns are nearly
% identical -- a 6 degree shift of a broad tuning curve barely moves
% anything. The difference is largest not at 90 degrees, where both
% stimuli drive the neuron equally hard, but on the FLANKS of the tuning
% curve, where the slope is steepest. This is the single most important
% intuition in neural decoding: information lives in the slope, not the
% peak.

% Homework question 1.
% (a) Sketch (or compute) how the difference curve in the bottom panel
%     changes as you increase dtheta from 2 to 45 degrees. At what
%     separation does the difference stop being well approximated by the
%     derivative of the tuning curve?
% (b) The difference curve has zero crossings. Where are they, and what
%     does a neuron sitting exactly at a zero crossing contribute to the
%     discrimination?
% (c) If you doubled kappa (sharper tuning), would the peak of the
%     difference curve get larger or smaller? Would it move?


%% ========================================================================
%  Part I.  One neuron: signal detection theory, ROC, and d-prime
%% ========================================================================

% Before touching the population, let us solve the problem for a single
% neuron, because every multivariate method in this tutorial reduces to
% this same problem after projecting onto one axis.
%
% Pick the neuron whose mean response is most ENHANCED by stimulus B. (The
% neuron most enhanced by A is equally informative; we take the B-preferring
% one only so that the ROC curve below bows above the diagonal, which is
% the conventional way to draw it. Part II returns to the sign issue.)

[~, best_neuron] = max(mu_B - mu_A);
fprintf('\nMost informative single neuron: #%d (preferred ori = %.1f deg)\n', ...
        best_neuron, pref_oris(best_neuron));

rA = X_A(:, best_neuron);    % its responses on stimulus-A trials
rB = X_B(:, best_neuron);    % its responses on stimulus-B trials

% ----- The two response distributions overlap -----
edges = linspace(min([rA;rB]), max([rA;rB]), 40);

figure(2); clf
subplot(2,2,1); hold on
histogram(rA, edges, 'FaceColor', [0.20 0.40 0.85], 'FaceAlpha', 0.5, 'EdgeColor','none')
histogram(rB, edges, 'FaceColor', [0.85 0.30 0.25], 'FaceAlpha', 0.5, 'EdgeColor','none')
xlabel('Response (spikes/s)'); ylabel('Number of trials')
legend(sprintf('%g deg', ori_A), sprintf('%g deg', ori_B), 'Location','northwest')
title(sprintf('Neuron %d: overlapping distributions', best_neuron))

% An ideal observer watching only this neuron must choose a CRITERION c and
% report "B" whenever the response exceeds c. Every choice of c trades off
% two quantities:
%
%   hit rate           P(report B | stimulus was B)
%   false alarm rate   P(report B | stimulus was A)
%
% A low criterion catches every B trial but also mislabels many A trials.
% A high criterion is conservative in both directions. Neither is "the"
% answer -- the full trade-off curve IS the answer, and that curve is the
% Receiver Operating Characteristic.

% ----- Illustrate three criteria -----
all_r = sort([rA; rB]);
crit_examples = all_r(round([0.25 0.5 0.75] * numel(all_r)));
subplot(2,2,1)
yl = ylim;
for c = crit_examples(:)'
    plot([c c], yl, 'k--', 'LineWidth', 1)
end

% ----- Build the ROC curve by hand -----
[pfa, phit, auc] = roc_curve(rA, rB);

subplot(2,2,2); hold on
plot(pfa, phit, 'k', 'LineWidth', 2)
plot([0 1], [0 1], ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1)
for c = crit_examples(:)'
    plot(mean(rA >= c), mean(rB >= c), 'o', 'MarkerSize', 8, ...
         'MarkerFaceColor', [0.95 0.75 0.20], 'MarkerEdgeColor','k')
end
axis square; axis([0 1 0 1])
xlabel('False alarm rate  P(report B | A)')
ylabel('Hit rate  P(report B | B)')
title(sprintf('ROC curve, AUC = %.3f', auc))

% The area under the ROC curve (AUC) summarizes the whole trade-off in one
% number. It has a beautiful interpretation that has nothing to do with
% criteria: AUC is exactly the probability that a randomly chosen B-trial
% response exceeds a randomly chosen A-trial response. That is precisely
% the performance of an ideal observer in a two-alternative forced choice
% task where both stimuli are presented and the observer picks the larger
% response. Let us verify this equivalence numerically.

auc_mw = mann_whitney_auc(rA, rB);
fprintf('AUC from trapezoidal integration : %.4f\n', auc);
fprintf('AUC from pairwise comparison     : %.4f\n', auc_mw);

% ----- d-prime: the parametric shortcut -----
% If both distributions are Gaussian with equal variance, the entire ROC is
% determined by a single number: the separation of the means in units of
% the common standard deviation.
%
%   d' = (mu_B - mu_A) / sqrt((var_A + var_B)/2)
%
% and the two summaries are linked by  AUC = normcdf(d'/sqrt(2)).

dprime = (mean(rB) - mean(rA)) / sqrt((var(rA) + var(rB))/2);
auc_from_dprime = normal_cdf(dprime/sqrt(2));

fprintf('d-prime                          : %.4f\n', dprime);
fprintf('AUC predicted from d-prime       : %.4f\n', auc_from_dprime);

% The agreement is good but not exact, because our responses are not
% Gaussian with equal variance -- variance scales with the mean, and the
% responses are clipped at zero. This is a general lesson: AUC is
% nonparametric and always valid, while d-prime buys you interpretability
% at the cost of a distributional assumption.

subplot(2,2,3); hold on
d_grid = linspace(0, 4, 200);
plot(d_grid, normal_cdf(d_grid/sqrt(2)), 'k', 'LineWidth', 2)
plot(dprime, auc, 'o', 'MarkerSize', 9, 'MarkerFaceColor',[0.85 0.30 0.25], ...
     'MarkerEdgeColor','k')
xlabel('d'''); ylabel('AUC')
title('AUC = \Phi(d''/\surd2) for equal-variance Gaussians')
ylim([0.5 1]); grid on

% ----- Criterion effects: the ROC is criterion-free, accuracy is not -----
crit_grid = linspace(min([rA;rB]), max([rA;rB]), 200);
acc_grid  = zeros(size(crit_grid));
for i = 1:numel(crit_grid)
    acc_grid(i) = ( mean(rB >= crit_grid(i)) + mean(rA < crit_grid(i)) ) / 2;
end

subplot(2,2,4); hold on
plot(crit_grid, acc_grid, 'k', 'LineWidth', 2)
plot([min(crit_grid) max(crit_grid)], [auc auc], '--', 'Color',[0.85 0.30 0.25])
xlabel('Criterion (spikes/s)'); ylabel('Balanced accuracy')
title('Accuracy depends on criterion; AUC does not')
legend('accuracy vs. criterion','AUC','Location','south')
ylim([0.4 1]); grid on

% Notice that the best accuracy achievable by any criterion is close to,
% but generally not equal to, the AUC. They coincide only in the
% equal-variance Gaussian case. Reporting AUC rather than "percent
% correct" is standard in neurophysiology precisely because it removes the
% experimenter's arbitrary choice of criterion.

% Homework question 2.
% (a) Show algebraically that AUC = P(r_B > r_A) by interpreting the ROC
%     integral as an expectation over criteria.
% (b) What is the AUC if the two distributions are identical? What is it
%     if neuron responses are LOWER for stimulus B than for A? Why do
%     neurophysiologists sometimes report max(AUC, 1-AUC)?
% (c) Set fano = 0 at the top of the script and re-run. What happens to
%     the ROC curve, and why is the resulting "perfect" performance
%     scientifically uninteresting?
% (d) The ROC curve above is jagged. What determines the size of the
%     steps, and how would you get a smooth curve?


%% ========================================================================
%  Part II.  Neurometric functions: which neurons carry the information?
%% ========================================================================

% We just chose the best neuron by looking at the difference of means. Now
% let us compute the AUC for EVERY neuron and see how discriminability is
% distributed across the population.

auc_all = zeros(N,1);
for i = 1:N
    auc_all(i) = mann_whitney_auc(X_A(:,i), X_B(:,i));
end

figure(3); clf
subplot(2,2,1); hold on
plot(pref_oris, auc_all, 'k', 'LineWidth', 1.5)
plot([0 180], [0.5 0.5], ':', 'Color',[0.5 0.5 0.5])
plot(pref_oris(best_neuron), auc_all(best_neuron), 'o', 'MarkerSize', 8, ...
     'MarkerFaceColor',[0.95 0.75 0.20], 'MarkerEdgeColor','k')
xlabel('Preferred orientation (deg)'); ylabel('AUC')
title('Single-neuron discriminability across the population')
xlim([0 180]); set(gca,'XTick',0:30:180)

% Two features deserve attention. First, the neurons tuned to exactly 90
% degrees -- the neurons that respond most strongly to both stimuli -- are
% the LEAST informative, with AUC near 0.5. Both stimuli sit symmetrically
% on their tuning peak, so their mean responses are equal. Second, AUC
% dips below 0.5 for half the population, which simply means those neurons
% fire MORE for stimulus A than for stimulus B. Information is information
% regardless of sign; a downstream decoder can flip the weight.

subplot(2,2,2); hold on
tc_slope = numerical_slope(pref_oris, kappa, theta_center, gain);
plot(pref_oris, abs(auc_all - 0.5)/max(abs(auc_all - 0.5)), 'k', 'LineWidth', 2)
plot(pref_oris, abs(tc_slope)/max(abs(tc_slope)), '--', 'LineWidth', 2, ...
     'Color', [0.85 0.30 0.25])
xlabel('Preferred orientation (deg)'); ylabel('Normalized magnitude')
legend('|AUC - 0.5|','|tuning-curve slope|','Location','north')
title('Discriminability tracks tuning-curve SLOPE')
xlim([0 180]); set(gca,'XTick',0:30:180)

% The overlay makes the point quantitatively: single-neuron
% discriminability for a fine discrimination is proportional to the
% derivative of the tuning curve at the reference orientation, divided by
% the noise standard deviation. Neurons at the peak have zero slope and
% contribute nothing. This is why "the neuron that responds most" and "the
% neuron that informs most" are different neurons.

% ----- The neurometric function -----
% Now sweep the difficulty of the task. For each angular separation, we
% ask how well the best single neuron performs, and how well the entire
% population performs when read out optimally. The resulting curves are
% NEUROMETRIC FUNCTIONS, directly comparable to the psychometric function
% measured behaviorally.

dtheta_grid = [1 2 3 4 6 8 12 16 24 32 45];
auc_best    = zeros(size(dtheta_grid));
auc_pop     = zeros(size(dtheta_grid));
n_tr_neuro  = 300;

for k = 1:numel(dtheta_grid)
    dA = theta_center - dtheta_grid(k)/2;
    dB = theta_center + dtheta_grid(k)/2;

    mA = tuning_mean(pref_oris, kappa, dA, baseline, gain);
    mB = tuning_mean(pref_oris, kappa, dB, baseline, gain);

    XA = sample_trials(mA, n_tr_neuro, fano, L_indep);
    XB = sample_trials(mB, n_tr_neuro, fano, L_indep);

    % best single neuron
    a = zeros(N,1);
    for i = 1:N
        a(i) = mann_whitney_auc(XA(:,i), XB(:,i));
    end
    auc_best(k) = max(max(a), 1-min(a));

    % population readout: project onto the difference-of-means axis
    w  = mB - mA;
    pA = XA * w;
    pB = XB * w;
    auc_pop(k) = mann_whitney_auc(pA, pB);
end

subplot(2,2,3); hold on
plot(dtheta_grid, auc_best, '-o', 'LineWidth', 2, 'Color',[0.20 0.40 0.85], ...
     'MarkerFaceColor',[0.20 0.40 0.85])
plot(dtheta_grid, auc_pop, '-s', 'LineWidth', 2, 'Color',[0.85 0.30 0.25], ...
     'MarkerFaceColor',[0.85 0.30 0.25])
plot([0 45], [0.5 0.5], ':', 'Color',[0.5 0.5 0.5])
set(gca,'XScale','log'); xlim([0.8 50]); ylim([0.45 1.02])
xlabel('Orientation separation (deg)'); ylabel('AUC')
legend('best single neuron','population readout','Location','southeast')
title('Neurometric functions')
grid on

% The population curve sits far to the left of the single-neuron curve: the
% population resolves separations that no individual neuron can. The gap
% between the curves is the value of pooling.

% ----- Threshold: the separation supporting 75% correct -----
thr_best = interp_threshold(dtheta_grid, auc_best, 0.75);
thr_pop  = interp_threshold(dtheta_grid, auc_pop,  0.75);
fprintf('\n75%% threshold, best single neuron : %.2f deg\n', thr_best);
fprintf('75%% threshold, population         : %.2f deg\n', thr_pop);

% ----- How does pooling scale with the number of neurons? -----
n_sub    = round(logspace(0, log10(N), 12));
auc_sub  = zeros(size(n_sub));
w_full   = mu_B - mu_A;
for k = 1:numel(n_sub)
    sel = round(linspace(1, N, n_sub(k)));      % evenly spread preferences
    pA  = X_A(:,sel) * w_full(sel);
    pB  = X_B(:,sel) * w_full(sel);
    auc_sub(k) = mann_whitney_auc(pA, pB);
end

subplot(2,2,4); hold on
plot(n_sub, auc_sub, '-o', 'LineWidth', 2, 'Color',[0.2 0.55 0.35], ...
     'MarkerFaceColor',[0.2 0.55 0.35])
plot([1 N], [0.5 0.5], ':', 'Color',[0.5 0.5 0.5])
set(gca,'XScale','log'); xlim([0.8 N*1.2]); ylim([0.45 1.02])
xlabel('Number of neurons pooled'); ylabel('AUC')
title('Pooling improves discriminability (independent noise)')
grid on

% Homework question 3.
% (a) With independent noise, d' for the pooled readout grows as sqrt(n).
%     Derive this. Does the curve above match that prediction? (Hint: plot
%     the equivalent d' rather than AUC.)
% (b) The best-neuron neurometric threshold is typically a few times worse
%     than the behavioral threshold of a trained animal, while the
%     population threshold is often far BETTER. What does that tell you
%     about the assumption of independent noise? (Part VII returns to this.)
% (c) Modify the code so that the population readout uses only neurons
%     preferring 60-70 degrees. How much does performance drop? Now use
%     only neurons preferring 88-92 degrees. Explain the difference.


%% ========================================================================
%  Part III.  Linear discriminant analysis
%% ========================================================================

% Every linear classifier has the same form: choose a weight vector w,
% project each trial's population response onto it, and compare the
% projection to a threshold.
%
%   decide B  if  w' * r + b > 0
%
% The methods differ only in how they choose w. In Part II we used the
% simplest possible choice, w = mu_B - mu_A, which points from one mean to
% the other. That is optimal when the noise is isotropic. When it is not,
% we can do better.
%
% Fisher's linear discriminant chooses w to maximize the separation of the
% projected means relative to the projected within-class variance:
%
%   w  proportional to  S^-1 (mu_B - mu_A)
%
% where S is the pooled within-class covariance matrix. The S^-1 term is
% doing something intuitive: it discounts directions in which the noise is
% large, and amplifies directions in which the noise is small.

% ----- Build intuition in two dimensions first -----
%
% A word about what follows. If we plot two neurons' responses to the 88
% and 92 degree stimuli, the two clouds sit almost exactly on top of each
% other. That is not a defect of the plot -- it is the whole point of the
% fine discrimination task. The best single neuron reaches an AUC of only
% about 0.6, so no PAIR of neurons can separate these stimuli either, and
% any boundary we draw would run through the middle of one indistinguishable
% blob. You cannot see a 4 degree effect by eye, which is exactly why the
% population analysis in the rest of this section is necessary.
%
% So for the two-dimensional PICTURES -- here and in the SVM section --
% we switch to a COARSER discrimination, 84 vs 96 degrees. That separation
% is chosen so the best available PAIR of neurons classifies at roughly 78
% percent: clearly better than chance, so the boundary is meaningful and
% you can see where it belongs, but far from perfect, so the two clouds
% still overlap and the picture stays honest about what real data looks
% like. The classifier mathematics is identical; only the difficulty
% changes. Every population result stays on the fine 88-vs-92 task.

ori_C = theta_center - 6;                % 84 deg, class 0
ori_D = theta_center + 6;                % 96 deg, class 1

mu_C = tuning_mean(pref_oris, kappa, ori_C, baseline, gain);
mu_D = tuning_mean(pref_oris, kappa, ori_D, baseline, gain);

% For these pictures we also add a shared GAIN fluctuation: on some trials
% the whole population is more responsive than on others (attention, arousal,
% state). This is the dominant mode of shared variability in cortex, and it
% is what makes the response clouds tilted rather than round. Part VII
% studies its consequences properly.
gain_sd = 0.25;
X_C = sample_trials(mu_C, n_trials, fano, L_indep) .* (1 + gain_sd*randn(n_trials,1));
X_D = sample_trials(mu_D, n_trials, fano, L_indep) .* (1 + gain_sd*randn(n_trials,1));
X_C = max(X_C, 0);
X_D = max(X_D, 0);

% For the coarse task, the two most useful neurons are simply the one that
% most prefers each stimulus.
[~, n1] = max(mu_D - mu_C);              % strongly prefers 120 deg
[~, n2] = min(mu_D - mu_C);              % strongly prefers 60 deg
pair = [n2 n1];                          % x axis = 60-preferring neuron

P_A = X_C(:, pair);
P_B = X_D(:, pair);

% How good is this pair, honestly? Cross-validate it.
acc_pair = cv_accuracy([P_A; P_B], [zeros(n_trials,1); ones(n_trials,1)], 5, ...
                       @(a,b,c) lda_predict(a, b, c, 0));

fprintf('\n2-D example (%g vs %g deg) uses neurons %d (%.1f deg) and %d (%.1f deg)\n', ...
        ori_C, ori_D, pair(1), pref_oris(pair(1)), pair(2), pref_oris(pair(2)));
fprintf('This pair classifies at %.1f%% (5-fold cross-validated)\n', 100*acc_pair);

[w_lda2, b_lda2] = lda_train(P_A, P_B, 0);
w_dom2 = (mean(P_B,1) - mean(P_A,1))';       % difference-of-means axis

figure(4); clf
subplot(1,2,1); hold on
plot(P_A(:,1), P_A(:,2), '.', 'MarkerSize', 6, 'Color',[0.20 0.40 0.85])
plot(P_B(:,1), P_B(:,2), '.', 'MarkerSize', 6, 'Color',[0.85 0.30 0.25])
plot(mean(P_A(:,1)), mean(P_A(:,2)), 'o','MarkerSize',10,'MarkerFaceColor',[0.10 0.20 0.55],'MarkerEdgeColor','k')
plot(mean(P_B(:,1)), mean(P_B(:,2)), 'o','MarkerSize',10,'MarkerFaceColor',[0.55 0.10 0.10],'MarkerEdgeColor','k')
draw_boundary(w_lda2, b_lda2, [P_A;P_B], 'k', '')
% Also draw the axis joining the two class means, for comparison
mid = (mean(P_A,1) + mean(P_B,1))/2;
arrow_scale = 6;
quiver(mid(1), mid(2), w_dom2(1)/norm(w_dom2)*arrow_scale, ...
       w_dom2(2)/norm(w_dom2)*arrow_scale, 0, 'LineWidth', 2, ...
       'Color', [0.95 0.65 0.15], 'MaxHeadSize', 1)
quiver(mid(1), mid(2), w_lda2(1)/norm(w_lda2)*arrow_scale, ...
       w_lda2(2)/norm(w_lda2)*arrow_scale, 0, 'LineWidth', 2, ...
       'Color', [0.1 0.1 0.1], 'MaxHeadSize', 1)
axis square
xlabel(sprintf('Neuron %d (pref %.0f deg)', pair(1), pref_oris(pair(1))))
ylabel(sprintf('Neuron %d (pref %.0f deg)', pair(2), pref_oris(pair(2))))
title(sprintf('Two neurons, two stimuli, one boundary (%.0f%%)', 100*acc_pair))
legend(sprintf('%g deg', ori_C), sprintf('%g deg', ori_D), 'Location','northeast')

% ----- Project onto the discriminant axis -----
proj_A = P_A * w_lda2;
proj_B = P_B * w_lda2;
[~,~,auc2] = roc_curve(proj_A, proj_B);

subplot(1,2,2); hold on
ed = linspace(min([proj_A;proj_B]), max([proj_A;proj_B]), 40);
histogram(proj_A, ed, 'FaceColor',[0.20 0.40 0.85],'FaceAlpha',0.5,'EdgeColor','none')
histogram(proj_B, ed, 'FaceColor',[0.85 0.30 0.25],'FaceAlpha',0.5,'EdgeColor','none')
xlabel('Projection onto LDA axis'); ylabel('Trials')
title(sprintf('After projection: a 1-D problem, AUC = %.3f', auc2))

% Now the geometry is legible: two tilted clouds, a boundary between them,
% and two arrows -- the LDA axis in black, the difference-of-means axis in
% orange. The clouds are tilted along the diagonal because the shared gain
% fluctuation pushes both neurons up and down together.
%
% Here the two arrows nearly coincide, and it is worth understanding why.
% The shared noise runs along the (+1,+1) diagonal, while the signal is
% OPPONENT -- stimulus D drives one neuron up and the other down, along
% (-1,+1). Signal and noise are close to orthogonal, so discounting the
% noise barely rotates the readout. Part VII constructs the opposite case,
% where the noise lies along the signal and the distinction between the
% two axes becomes the whole story.
%
% Notice also what just happened. A two-dimensional classification problem
% became the one-dimensional signal detection problem of Part I. This is
% true of every linear classifier and it is why Part I was worth the time:
% ROC, AUC, and d-prime apply unchanged to the projected variable.

% ----- Now the full population -----
% With N = 80 neurons and 400 trials per class, we can still estimate the
% covariance matrix, but only barely. Let us look at how well conditioned
% it is.

S_pool = pooled_cov(X_A, X_B);
ev = sort(eig(S_pool), 'descend');
fprintf('Covariance matrix: %d x %d, condition number = %.3g\n', ...
        N, N, max(ev)/max(min(ev), eps));

figure(5); clf
subplot(2,2,1)
semilogy(max(ev, 1e-12), 'k', 'LineWidth', 1.5)
xlabel('Eigenvalue index'); ylabel('Eigenvalue')
title('Spectrum of the within-class covariance')
grid on

% If the number of trials per class is smaller than the number of neurons,
% S is singular and S^-1 does not exist. Even when it is invertible, the
% smallest eigenvalues are estimated terribly, and inverting the matrix
% amplifies exactly those unreliable directions. The standard fix is
% SHRINKAGE regularization:
%
%   S_reg = (1 - lambda) * S + lambda * mean(diag(S)) * I
%
% which pulls the estimate toward a scaled identity matrix. lambda = 0 is
% plain LDA; lambda = 1 ignores covariance entirely and reduces to the
% difference-of-means decoder.

lambda_lda = 0.05;
[w_lda, b_lda] = lda_train(X_A, X_B, lambda_lda);
w_dom = mu_B - mu_A;

sc_lda_A = X_A * w_lda;   sc_lda_B = X_B * w_lda;
sc_dom_A = X_A * w_dom;   sc_dom_B = X_B * w_dom;

auc_lda = mann_whitney_auc(sc_lda_A, sc_lda_B);
auc_dom = mann_whitney_auc(sc_dom_A, sc_dom_B);

fprintf('Training-set AUC, regularized LDA      : %.4f\n', auc_lda);
fprintf('Training-set AUC, difference-of-means  : %.4f\n', auc_dom);

subplot(2,2,2); hold on
plot(pref_oris, w_lda/norm(w_lda), 'k', 'LineWidth', 1.5)
plot(pref_oris, w_dom/norm(w_dom), '--', 'LineWidth', 2, 'Color',[0.85 0.30 0.25])
plot([0 180],[0 0],':','Color',[0.5 0.5 0.5])
xlabel('Preferred orientation (deg)'); ylabel('Normalized weight')
legend('LDA','difference of means','Location','northeast')
title('Decoder weights as a function of preference')
xlim([0 180]); set(gca,'XTick',0:30:180)

% The weight profile is the derivative of the tuning curve, with opposite
% signs on the two flanks. The decoder is implementing exactly the
% opponent comparison that the neurometric analysis predicted: subtract
% the neurons that prefer A from the neurons that prefer B.

subplot(2,2,3); hold on
ed = linspace(min([sc_lda_A;sc_lda_B]), max([sc_lda_A;sc_lda_B]), 40);
histogram(sc_lda_A, ed, 'FaceColor',[0.20 0.40 0.85],'FaceAlpha',0.5,'EdgeColor','none')
histogram(sc_lda_B, ed, 'FaceColor',[0.85 0.30 0.25],'FaceAlpha',0.5,'EdgeColor','none')
xlabel('LDA projection'); ylabel('Trials')
title(sprintf('Population LDA, AUC = %.3f', auc_lda))

% ----- How much regularization? -----
lam_grid = [0 logspace(-4, 0, 25)];
auc_train = zeros(size(lam_grid));
auc_test  = zeros(size(lam_grid));

% Hold out half the trials to get an honest estimate
hold_idx = 1:2:n_trials;
trn_idx  = 2:2:n_trials;
for k = 1:numel(lam_grid)
    [wk, ~] = lda_train(X_A(trn_idx,:), X_B(trn_idx,:), lam_grid(k));
    auc_train(k) = mann_whitney_auc(X_A(trn_idx,:)*wk, X_B(trn_idx,:)*wk);
    auc_test(k)  = mann_whitney_auc(X_A(hold_idx,:)*wk, X_B(hold_idx,:)*wk);
end

subplot(2,2,4); hold on
plot(lam_grid, auc_train, '-o','LineWidth',1.8,'Color',[0.5 0.5 0.5],'MarkerSize',4)
plot(lam_grid, auc_test,  '-o','LineWidth',1.8,'Color',[0.85 0.30 0.25],'MarkerSize',4)
set(gca,'XScale','log'); xlim([1e-4 1])
xlabel('Shrinkage \lambda'); ylabel('AUC')
legend('training','held out','Location','southwest')
title('Regularization closes the train/test gap')
grid on

% The training curve is monotonically decreasing -- less regularization
% always fits the training data better. The held-out curve is
% non-monotonic, with a peak at intermediate lambda. The vertical gap
% between the curves is overfitting made visible. Part VI develops this
% properly with cross-validation.

% Homework question 4.
% (a) Derive w proportional to S^-1 (mu_B - mu_A) by maximizing the Fisher
%     criterion  (w'(mu_B-mu_A))^2 / (w' S w)  with respect to w.
% (b) Show that when S is proportional to the identity, LDA and the
%     difference-of-means decoder give the same boundary.
% (c) Reduce n_trials to 50 and re-run this section. What happens to the
%     condition number, to the unregularized LDA weights, and to the
%     optimal lambda?
% (d) LDA assumes both classes share one covariance matrix. In our
%     simulation variance scales with the mean, so the two classes have
%     slightly different covariances. What classifier would you use if the
%     difference were large? (Look up quadratic discriminant analysis.)


%% ========================================================================
%  Part IV.  Logistic regression: a GLM classifier
%% ========================================================================

% LDA is a GENERATIVE method: it models the response distribution for each
% class and derives a boundary. Logistic regression is DISCRIMINATIVE: it
% models the class probability directly and never describes the responses
% themselves.
%
%   P(y = 1 | r) = 1 / (1 + exp(-(w'r + b)))
%
% Equivalently, the log-odds of class B are linear in the population
% response. This is a generalized linear model with a Bernoulli noise model
% and a logit link -- the same GLM machinery used to fit spike-count data,
% pointed at a binary outcome instead.
%
% We fit by maximizing the penalized log-likelihood
%
%   sum_t [ y_t * eta_t - log(1 + exp(eta_t)) ] - (lambda/2) * ||w||^2
%
% using iteratively reweighted least squares (Newton's method). The ridge
% penalty is not optional here: with N = 80 correlated predictors and
% perfectly separable training data, the unpenalized likelihood is
% maximized by weights of infinite magnitude.

lambda_lr = 1.0;
w_lr_full = logistic_train(X, y, lambda_lr);
b_lr = w_lr_full(1);
w_lr = w_lr_full(2:end);

p_hat = logistic_predict(X, w_lr_full);
auc_lr = mann_whitney_auc(p_hat(y==0), p_hat(y==1));
fprintf('\nTraining-set AUC, ridge logistic regression : %.4f\n', auc_lr);

figure(6); clf
subplot(2,2,1); hold on
plot(pref_oris, w_lr/norm(w_lr), 'LineWidth', 1.8, 'Color',[0.35 0.25 0.65])
plot(pref_oris, w_lda/norm(w_lda), 'k--', 'LineWidth', 1.5)
plot([0 180],[0 0],':','Color',[0.5 0.5 0.5])
xlabel('Preferred orientation (deg)'); ylabel('Normalized weight')
legend('logistic','LDA','Location','northeast')
title('Logistic and LDA weights are nearly proportional')
xlim([0 180]); set(gca,'XTick',0:30:180)

cos_ang = (w_lr'*w_lda) / (norm(w_lr)*norm(w_lda));
fprintf('Angle between logistic and LDA weight vectors: %.1f deg\n', ...
        acos(min(max(cos_ang,-1),1))*180/pi);

% They agree closely, and this is not a coincidence. If the two classes
% really are Gaussian with a shared covariance, logistic regression and LDA
% estimate the same population boundary; they differ only in what they
% assume and therefore in how they behave when the assumption fails.
% Logistic regression is more robust to non-Gaussian responses and to
% outliers; LDA is more efficient when its assumptions hold, which matters
% when trials are scarce.

% ----- The output is a calibrated probability, not just a label -----
subplot(2,2,2); hold on
ed = linspace(0,1,30);
histogram(p_hat(y==0), ed, 'FaceColor',[0.20 0.40 0.85],'FaceAlpha',0.5,'EdgeColor','none')
histogram(p_hat(y==1), ed, 'FaceColor',[0.85 0.30 0.25],'FaceAlpha',0.5,'EdgeColor','none')
xlabel('Predicted P(stimulus = B)'); ylabel('Trials')
title('Logistic outputs are probabilities')

% ----- Check calibration -----
% A well-calibrated model has the property that among trials where it says
% "70% chance of B", about 70% really are B.
[bin_c, bin_obs] = calibration_curve(p_hat, y, 10);
subplot(2,2,3); hold on
plot([0 1],[0 1],':','Color',[0.5 0.5 0.5],'LineWidth',1.5)
plot(bin_c, bin_obs, '-o','LineWidth',2,'Color',[0.35 0.25 0.65],'MarkerFaceColor',[0.35 0.25 0.65])
axis square; axis([0 1 0 1])
xlabel('Predicted probability'); ylabel('Observed fraction of class B')
title('Calibration')
grid on

% This probabilistic output is the practical advantage of the GLM. It lets
% you ask questions that a bare label cannot answer: on which trials was
% the population pattern ambiguous, and does the animal's behavior on those
% trials track the ambiguity? That analysis -- relating decoder confidence
% to choice -- is a standard tool in perceptual decision-making work.

% ----- Regularization path -----
lam_grid_lr = logspace(-2, 3, 20);
w_path = zeros(numel(lam_grid_lr), N);
for k = 1:numel(lam_grid_lr)
    wk = logistic_train(X(trn_idx_full(n_trials),:), y(trn_idx_full(n_trials)), lam_grid_lr(k));
    w_path(k,:) = wk(2:end)';
end

subplot(2,2,4); hold on
plot(lam_grid_lr, w_path(:, round(linspace(1,N,15))), 'LineWidth', 1.2)
set(gca,'XScale','log')
xlabel('Ridge penalty \lambda'); ylabel('Weight')
title('Regularization path (15 example neurons)')
grid on

% As lambda grows, all weights shrink smoothly toward zero. Ridge (L2)
% shrinks but never zeroes them. If you replaced the L2 penalty with L1
% (lasso), weights would hit exactly zero one by one, giving a sparse
% decoder that reads out only a handful of neurons -- often a more
% biologically plausible hypothesis, and easier to interpret.

% Homework question 5.
% (a) Write down the gradient and Hessian of the penalized log-likelihood
%     and confirm they match the IRLS update in logistic_train below.
% (b) Set lambda_lr = 0 and n_trials = 30, then re-fit. What happens to
%     the magnitude of the weights, and why? (This is called complete
%     separation.)
% (c) The logistic model assumes log-odds are LINEAR in firing rate. Name
%     a situation in neural data where that is clearly wrong, and describe
%     how you would modify the model.
% (d) Implement an L1 penalty by proximal gradient descent and compare the
%     resulting weight profile to the ridge solution.


%% ========================================================================
%  Part V.  Support vector machines and the margin
%% ========================================================================

% LDA asks about distributions. Logistic regression asks about
% probabilities. The support vector machine asks a purely geometric
% question: of all the hyperplanes that separate the two classes, which one
% sits as far as possible from the nearest points of either class?
%
% That distance is the MARGIN, and the trials that touch it are the SUPPORT
% VECTORS. Everything else in the data set is irrelevant to the solution --
% you could delete it and get the same boundary. This is very different
% from LDA, where every trial contributes to the mean and covariance.
%
% Real neural data is never perfectly separable, so we use the SOFT-MARGIN
% formulation: minimize
%
%   (1/2)||w||^2 + C * sum_t max(0, 1 - y_t (w'r_t + b))
%
% The hinge loss penalizes points on the wrong side of the margin. C
% controls the trade-off; small C means a wide margin with many violations
% (heavily regularized), large C means a narrow margin that respects the
% training data (lightly regularized). C is the inverse of a
% regularization strength, so it moves opposite to lambda.

has_svm = exist('fitcsvm','file') == 2;
if has_svm
    fprintf('\nfitcsvm is available; using the Statistics Toolbox SVM.\n');
else
    fprintf('\nfitcsvm not found; using the hand-written hinge-loss classifier.\n');
end

% ----- 2-D illustration so the margin is visible -----
P2 = [P_A; P_B];
y2 = [zeros(size(P_A,1),1); ones(size(P_B,1),1)];

figure(7); clf
C_list = [0.01 1];
for ci = 1:2
    subplot(1,3,ci); hold on
    plot(P_A(:,1), P_A(:,2), '.', 'MarkerSize', 6, 'Color',[0.20 0.40 0.85])
    plot(P_B(:,1), P_B(:,2), '.', 'MarkerSize', 6, 'Color',[0.85 0.30 0.25])

    if has_svm
        mdl = fitcsvm(P2, y2, 'KernelFunction','linear', ...
                      'BoxConstraint', C_list(ci), 'Standardize', true);
        % Undo standardization so the weights live in raw response units
        wv = mdl.Beta ./ mdl.Sigma(:);
        bv = mdl.Bias - sum(mdl.Beta .* mdl.Mu(:) ./ mdl.Sigma(:));
        sv = mdl.SupportVectors .* mdl.Sigma + mdl.Mu;
        plot(sv(:,1), sv(:,2), 'o', 'MarkerSize', 7, 'Color',[0.1 0.1 0.1], 'LineWidth', 0.8)
        n_sv = size(sv,1);
    else
        [wv, bv] = hinge_train(P2, y2, C_list(ci));
        margin = y2*2-1;
        f = P2*wv + bv;
        on_margin = (margin .* f) < 1.05;
        plot(P2(on_margin,1), P2(on_margin,2), 'o','MarkerSize',7, ...
             'Color',[0.1 0.1 0.1],'LineWidth',0.8)
        n_sv = sum(on_margin);
    end

    draw_boundary(wv, bv, P2, 'k', '');
    draw_boundary(wv, bv - 1, P2, [0.45 0.45 0.45], '');
    draw_boundary(wv, bv + 1, P2, [0.45 0.45 0.45], '');
    axis square
    xlabel(sprintf('Neuron %d', n1)); ylabel(sprintf('Neuron %d', n2))
    title(sprintf('SVM, C = %g  (%d support vectors)', C_list(ci), n_sv))
end

% Circled points are support vectors. With small C the margin is wide and
% most of the data lies inside it, so almost every trial is a support
% vector and the boundary is determined by the bulk of the data. With
% large C the margin is narrow, few trials define the boundary, and the
% solution is more sensitive to individual noisy trials.

% ----- Linear kernel is not always enough -----
% Consider a genuinely nonlinear problem: discriminate CARDINAL
% orientations (0 and 90) from OBLIQUE ones (45 and 135), reading out from
% two neurons preferring 22.5 and 67.5 degrees. Those preferences are
% chosen to sit exactly between the stimuli, and the consequence is worth
% working out on paper before you look at the plot:
%
%       stimulus     neuron 22.5     neuron 67.5
%          0 deg        high            low
%         90 deg        low             high
%         45 deg        high            high
%        135 deg        low             low
%
% Each neuron alone is completely uninformative about the category: neuron
% 22.5 is high for one cardinal (0) and one oblique (45). The category is
% the EXCLUSIVE OR of the two responses, and the four clusters sit at the
% four corners of a square with the two classes on opposite diagonals.
% This is the canonical problem that no straight line can solve, and it is
% not a contrived one -- it is what "the information is present but not
% linearly available" actually looks like.
%
% We use a higher-contrast stimulus for this panel so the four clusters
% separate cleanly; the geometry is the point, not the difficulty.

oris_card = [0 90];
oris_obl  = [45 135];
n_tr_k    = 250;
gain_hi   = 20;                          % higher contrast for this example

Xc = []; Xo = [];
for o = oris_card
    Xc = [Xc; sample_trials(tuning_mean(pref_oris,kappa,o,baseline,gain_hi), n_tr_k, fano, L_indep)];
end
for o = oris_obl
    Xo = [Xo; sample_trials(tuning_mean(pref_oris,kappa,o,baseline,gain_hi), n_tr_k, fano, L_indep)];
end

pair_k = [find_pref(pref_oris, 22.5), find_pref(pref_oris, 67.5)];
K2 = [Xc(:,pair_k); Xo(:,pair_k)];
yk = [zeros(size(Xc,1),1); ones(size(Xo,1),1)];

% Honest accuracies: fit on half the trials, score on the other half. The
% training-set accuracy of an RBF SVM is nearly meaningless, since a narrow
% kernel can memorize any data set.
k_folds = make_folds(numel(yk), 2);
k_tr = k_folds == 1;  k_te = ~k_tr;

subplot(1,3,3); hold on
plot(K2(yk==0,1), K2(yk==0,2), '.', 'MarkerSize', 5, 'Color',[0.20 0.40 0.85])
plot(K2(yk==1,1), K2(yk==1,2), '.', 'MarkerSize', 5, 'Color',[0.85 0.30 0.25])

if has_svm
    mdl_lin = fitcsvm(K2(k_tr,:), yk(k_tr), 'KernelFunction','linear', ...
                      'Standardize',true);
    mdl_rbf = fitcsvm(K2(k_tr,:), yk(k_tr), 'KernelFunction','rbf', ...
                      'KernelScale', 2, 'BoxConstraint', 1, 'Standardize', true);
    acc_lin = mean(predict(mdl_lin, K2(k_te,:)) == yk(k_te));
    acc_rbf = mean(predict(mdl_rbf, K2(k_te,:)) == yk(k_te));

    % Draw the RBF decision surface
    gx = linspace(min(K2(:,1)), max(K2(:,1)), 150);
    gy = linspace(min(K2(:,2)), max(K2(:,2)), 150);
    [GX, GY] = meshgrid(gx, gy);
    [~, sc] = predict(mdl_rbf, [GX(:) GY(:)]);
    contour(GX, GY, reshape(sc(:,2), size(GX)), [0 0], 'k', 'LineWidth', 2)
    title(sprintf('Cardinal vs oblique (held out): linear %.2f, RBF %.2f', ...
                  acc_lin, acc_rbf))
else
    [wk_, bk_] = hinge_train(K2(k_tr,:), yk(k_tr), 1);
    acc_lin = mean(((K2(k_te,:)*wk_ + bk_) > 0) == yk(k_te));
    title(sprintf('Cardinal vs oblique (held out): linear %.2f -- RBF needs the Stats Toolbox', ...
                  acc_lin))
end
xlabel(sprintf('Neuron %d (pref %.0f deg)', pair_k(1), pref_oris(pair_k(1))))
ylabel(sprintf('Neuron %d (pref %.0f deg)', pair_k(2), pref_oris(pair_k(2))))
axis square

fprintf('Cardinal vs oblique, held-out accuracy: linear SVM %.3f\n', acc_lin);
if has_svm
    fprintf('Cardinal vs oblique, held-out accuracy: RBF SVM    %.3f\n', acc_rbf);
end

% The linear SVM sits at chance, as it must: no line separates opposite
% corners of a square from the other two. The RBF kernel implicitly maps
% each trial into a very high-dimensional feature space where the two
% classes DO become linearly separable, then finds a maximum-margin
% hyperplane there. Projected back down, that hyperplane becomes the closed
% contours you see around the two cardinal clusters. The kernel trick means
% you never compute the mapping -- you only ever evaluate inner products
% k(r, r') = exp(-||r - r'||^2 / (2*sigma^2)).
%
% Note that both accuracies here are measured on held-out trials. The
% training-set accuracy of an RBF SVM is close to meaningless: with a
% narrow enough kernel it can place a small bubble around every training
% point and score 100 percent while learning nothing.
%
% A word of caution before you reach for the kernel in real analyses. A
% nonlinear classifier that beats a linear one tells you that the
% information is present but not linearly available. Whether a downstream
% neuron could actually extract it is a separate question, and most
% readout hypotheses in systems neuroscience are deliberately linear for
% exactly that reason.

% Homework question 6.
% (a) Explain why deleting all non-support-vector trials leaves the SVM
%     solution unchanged, and why the same is not true for LDA.
% (b) The hinge loss and the logistic loss both penalize misclassification.
%     Plot both as a function of the margin y*f(r) and describe how they
%     differ for confidently correct and grossly misclassified points.
% (c) Vary KernelScale for the RBF SVM over several orders of magnitude.
%     What happens to the decision boundary at very small and very large
%     values? Which failure mode is overfitting?
% (d) Construct a version of the cardinal-vs-oblique task that IS linearly
%     separable by choosing different neurons -- try neurons preferring 0
%     and 90 degrees instead of 22.5 and 67.5, and work out from the tuning
%     curves why a straight line suddenly suffices. What does this tell you
%     about the claim that a category is "nonlinearly encoded"?


%% ========================================================================
%  Part VI.  Cross-validation, overfitting, and regularization
%% ========================================================================

% Every accuracy we have reported so far was measured on the same trials
% used to fit the classifier. Those numbers are not estimates of anything
% you care about. A classifier with enough free parameters can memorize
% noise, and with N = 80 neurons and as few as 5 training trials per
% class it has plenty of freedom to do so.
%
% The remedy is to evaluate on data the fit never saw. We implement k-fold
% cross-validation directly rather than calling a toolbox routine, because
% the procedure is short and worth seeing.

kfold = 10;

% Draw the folds ONCE and reuse them for every method. Comparing two
% classifiers on different random splits adds noise that has nothing to do
% with the classifiers.
folds = make_folds(numel(y), kfold);

acc_lda_cv = cv_accuracy(X, y, folds, @(Xtr,ytr,Xte) ...
    lda_predict(Xtr, ytr, Xte, 0.05));
acc_dom_cv = cv_accuracy(X, y, folds, @(Xtr,ytr,Xte) ...
    lda_predict(Xtr, ytr, Xte, 1.0));
acc_lr_cv  = cv_accuracy(X, y, folds, @(Xtr,ytr,Xte) ...
    logistic_predict(Xte, logistic_train(Xtr, ytr, 1.0)) > 0.5);

fprintf('\n%d-fold cross-validated accuracy:\n', kfold);
fprintf('  regularized LDA        : %.3f\n', acc_lda_cv);
fprintf('  difference of means    : %.3f\n', acc_dom_cv);
fprintf('  ridge logistic         : %.3f\n', acc_lr_cv);

% ----- Overfitting made explicit: vary the number of training trials -----
% The last 100 trials of each class are reserved as a fixed test set and
% never used for fitting, so the training pool is the first 300.
test_idx     = (n_trials-99):n_trials;
n_train_grid = [5 10 20 30 40 60 80 120 200 300];
acc_tr = zeros(size(n_train_grid));
acc_te = zeros(size(n_train_grid));

for k = 1:numel(n_train_grid)
    nt = n_train_grid(k);
    trA = X_A(1:nt, :);  trB = X_B(1:nt, :);
    teA = X_A(test_idx, :);  teB = X_B(test_idx, :);

    [wk, bk] = lda_train(trA, trB, 0.0);         % NO regularization
    acc_tr(k) = 0.5*(mean(trA*wk + bk < 0) + mean(trB*wk + bk > 0));
    acc_te(k) = 0.5*(mean(teA*wk + bk < 0) + mean(teB*wk + bk > 0));
end

figure(8); clf
subplot(2,2,1); hold on
plot(n_train_grid, acc_tr, '-o','LineWidth',2,'Color',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
plot(n_train_grid, acc_te, '-o','LineWidth',2,'Color',[0.85 0.30 0.25],'MarkerFaceColor',[0.85 0.30 0.25])
plot([min(n_train_grid) max(n_train_grid)], [0.5 0.5], ':','Color',[0.5 0.5 0.5])
set(gca,'XScale','log')
xlabel('Training trials per class'); ylabel('Accuracy')
legend('training','test','Location','east')
title(sprintf('Learning curve, unregularized LDA (N = %d)', N))
ylim([0.3 1.05]); grid on

% This is the single most important figure in the tutorial. When training
% trials are fewer than neurons, unregularized LDA classifies the training
% data almost perfectly while performing at or near chance on held-out
% trials. It has found a direction that separates these particular noise
% samples, and that direction means nothing. Only once the training set
% comfortably exceeds the number of neurons do the two curves converge,
% and they converge from opposite directions: training accuracy falls as
% test accuracy rises.
%
% If you have ever seen a decoding result reported without
% cross-validation, this figure is why you should not believe it.

% ----- The same experiment with regularization -----
acc_te_reg = zeros(size(n_train_grid));
for k = 1:numel(n_train_grid)
    nt = n_train_grid(k);
    trA = X_A(1:nt, :);  trB = X_B(1:nt, :);
    teA = X_A(test_idx, :);  teB = X_B(test_idx, :);
    [wk, bk] = lda_train(trA, trB, 0.2);         % shrinkage
    acc_te_reg(k) = 0.5*(mean(teA*wk + bk < 0) + mean(teB*wk + bk > 0));
end

subplot(2,2,2); hold on
plot(n_train_grid, acc_te, '-o','LineWidth',2,'Color',[0.85 0.30 0.25],'MarkerFaceColor',[0.85 0.30 0.25])
plot(n_train_grid, acc_te_reg, '-s','LineWidth',2,'Color',[0.2 0.55 0.35],'MarkerFaceColor',[0.2 0.55 0.35])
plot([min(n_train_grid) max(n_train_grid)], [0.5 0.5], ':','Color',[0.5 0.5 0.5])
set(gca,'XScale','log')
xlabel('Training trials per class'); ylabel('Test accuracy')
legend('\lambda = 0','\lambda = 0.2','Location','southeast')
title('Regularization rescues the small-sample regime')
ylim([0.3 1.05]); grid on

% ----- Choosing lambda properly: nested cross-validation -----
% The temptation is to pick the lambda that maximizes cross-validated
% accuracy and then report that accuracy. That is a subtle form of
% overfitting: you used the test folds to make a modeling decision. The
% correct procedure nests an inner cross-validation loop inside the outer
% one.

lam_cv = [0 0.001 0.01 0.05 0.1 0.2 0.5 1];
acc_by_lam = zeros(size(lam_cv));
for k = 1:numel(lam_cv)
    acc_by_lam(k) = cv_accuracy(X, y, folds, @(Xtr,ytr,Xte) ...
        lda_predict(Xtr, ytr, Xte, lam_cv(k)));
end

subplot(2,2,3); hold on
plot(lam_cv, acc_by_lam, '-o','LineWidth',2,'Color',[0.35 0.25 0.65],'MarkerFaceColor',[0.35 0.25 0.65])
xlabel('Shrinkage \lambda'); ylabel('CV accuracy')
title('Cross-validated accuracy vs. regularization')
grid on

acc_nested = nested_cv_accuracy(X, y, 5, lam_cv);
[~, ibest] = max(acc_by_lam);
fprintf('\nBest single-loop CV accuracy (lambda = %g) : %.3f  <- optimistic\n', ...
        lam_cv(ibest), acc_by_lam(ibest));
fprintf('Nested CV accuracy                          : %.3f  <- honest\n', acc_nested);

% ----- A null distribution: how good is "good"? -----
% An accuracy of 0.62 sounds unimpressive but may be highly significant
% with enough trials. The way to find out is a permutation test: shuffle
% the labels, re-run the whole cross-validated pipeline, and repeat.

n_perm = 50;
acc_null = zeros(n_perm,1);
for p = 1:n_perm
    y_shuf = y(randperm(numel(y)));
    acc_null(p) = cv_accuracy(X, y_shuf, 5, @(Xtr,ytr,Xte) ...
        lda_predict(Xtr, ytr, Xte, 0.05));
end
p_value = (1 + sum(acc_null >= acc_lda_cv)) / (1 + n_perm);

subplot(2,2,4); hold on
histogram(acc_null, 15, 'FaceColor',[0.6 0.6 0.6], 'EdgeColor','none')
yl = ylim;
plot([acc_lda_cv acc_lda_cv], yl, 'r', 'LineWidth', 2)
xlabel('CV accuracy'); ylabel('Count')
title(sprintf('Label-shuffled null,  p < %.3f', p_value))
legend('shuffled labels','observed','Location','northwest')

% Note that the null distribution is centered on 0.5 but has real width.
% With few trials that width can be large, and a decoder that looks
% impressive may fall well inside it.

% Homework question 7.
% (a) Why does the unregularized learning curve show test accuracy BELOW
%     chance for the smallest training sets? What would you expect if you
%     averaged over many random draws of the training set?
% (b) Explain in your own words why the single-loop CV number in the
%     printout is optimistic. How large is the bias here, and when would
%     you expect it to be worse?
% (c) Our cross-validation shuffles trials randomly. In a real experiment
%     with slow drift in the recording, why might that be too permissive,
%     and what would you do instead?
% (d) Increase n_perm to 500 and report an exact p-value. How does the
%     width of the null distribution depend on the number of trials?


%% ========================================================================
%  Part VII.  Decoding geometry: signal vs. noise correlations
%% ========================================================================

% Everything so far assumed independent noise. Real cortical neurons are
% correlated: pairs with similar preferred orientations share more noise
% than pairs with different preferences. This section shows that whether
% correlations help or hurt depends entirely on their GEOMETRY relative to
% the signal.
%
% We generate limited-range correlations,
%
%   c_ij = c0 * exp(-|phi_i - phi_j| / tau)
%
% and compare decoding with and without them.

% Note on c0: measured spike-count correlations in visual cortex are
% usually 0.1 to 0.2 for nearby, similarly tuned pairs. We use a larger
% value here so that the effect is visible in a single simulation rather
% than only in an average over many. Homework question 8 asks you to
% repeat this with realistic values.
c0  = 0.5;       % peak pairwise correlation
tau = 25;        % correlation length in degrees

C_lr = limited_range_corr(pref_oris, c0, tau);
L_lr = chol(nearest_pd(C_lr));

XA_c = sample_trials(mu_A, n_trials, fano, L_lr);
XB_c = sample_trials(mu_B, n_trials, fano, L_lr);

figure(9); clf
subplot(2,3,1)
imagesc(pref_oris, pref_oris, C_lr); axis square; colorbar
xlabel('Preferred ori (deg)'); ylabel('Preferred ori (deg)')
title('Noise correlation matrix')

% ----- Compare the two decoders under the two noise structures -----
w_dom_n = (mu_B - mu_A) / norm(mu_B - mu_A);

d_ind_dom = dprime_along(X_A,  X_B,  w_dom_n);
d_cor_dom = dprime_along(XA_c, XB_c, w_dom_n);

[w_i, ~] = lda_train(X_A,  X_B,  0.05);
[w_c, ~] = lda_train(XA_c, XB_c, 0.05);
d_ind_opt = dprime_along(X_A,  X_B,  w_i/norm(w_i));
d_cor_opt = dprime_along(XA_c, XB_c, w_c/norm(w_c));

subplot(2,3,2)
bar([d_ind_dom d_cor_dom; d_ind_opt d_cor_opt]')
set(gca,'XTickLabel',{'independent','correlated'})
ylabel('d''')
legend('difference-of-means','LDA (optimal)','Location','northeast')
title('Correlations penalize the naive decoder')
grid on

% Two things happen when correlations are introduced. First, both decoders
% lose d-prime: shared noise is not averaged away by pooling, so the
% population is simply less informative than an independent one with the
% same tuning. Second, and more interesting, the ADVANTAGE of the optimal
% decoder over the naive one grows. With independent noise, knowing the
% covariance buys you only a few percent; with strong correlations, the
% S^-1 term earns its keep by steering the readout away from the noisy
% directions. Print the ratio and see for yourself:

fprintf('\nOptimal / naive d-prime, independent noise : %.3f\n', d_ind_opt/d_ind_dom);
fprintf('Optimal / naive d-prime, correlated noise  : %.3f\n', d_cor_opt/d_cor_dom);

% This is why "does the decoder account for noise correlations?" is a real
% experimental question and not a technicality. But notice that the loss
% here is a matter of degree. The next simulation shows a case where the
% loss is absolute.

% ----- The critical case: information-limiting correlations -----
% The geometry that truly matters is whether noise lies ALONG the signal
% direction. Noise in any other direction can be projected out by a
% suitable decoder. Noise parallel to mu_B - mu_A cannot be, no matter how
% many neurons you record, because it is indistinguishable from a change in
% the stimulus.
%
% We add a rank-one noise component along the signal axis and sweep its
% amplitude.

f_prime = (mu_B - mu_A) / norm(mu_B - mu_A);
eps_grid = [0 logspace(-3, 0, 12)];
d_opt_grid = zeros(size(eps_grid));
d_opt_n    = zeros(size(eps_grid));

S_base = diag(fano * (mu_A + mu_B)/2);
for k = 1:numel(eps_grid)
    S_k = S_base + eps_grid(k) * norm(mu_B-mu_A)^2 * (f_prime * f_prime');
    % Optimal (analytic) discriminability:  d'^2 = df' * S^-1 * df
    df = mu_B - mu_A;
    d_opt_grid(k) = sqrt(df' * (S_k \ df));

    % Same, but with 4x as many neurons (subsample-and-scale surrogate)
    S_k2 = S_base/4 + eps_grid(k) * norm(mu_B-mu_A)^2 * (f_prime * f_prime');
    d_opt_n(k) = sqrt(df' * (S_k2 \ df));
end

subplot(2,3,3); hold on
plot(eps_grid, d_opt_grid, '-o','LineWidth',2,'Color',[0.20 0.40 0.85],'MarkerFaceColor',[0.20 0.40 0.85])
plot(eps_grid, d_opt_n, '-s','LineWidth',2,'Color',[0.85 0.30 0.25],'MarkerFaceColor',[0.85 0.30 0.25])
set(gca,'XScale','log'); set(gca,'YScale','log')
xlabel('Amplitude of signal-aligned noise \epsilon'); ylabel('Optimal d''')
legend('N neurons','4N neurons','Location','southwest')
title('Information-limiting correlations saturate')
grid on

% With no signal-aligned noise, quadrupling the population doubles d-prime,
% exactly as independent pooling predicts. With signal-aligned noise, the
% two curves converge: adding neurons stops helping. The population has an
% information ceiling that no amount of additional recording can raise.
% This is the central result of Moreno-Bote et al. (2014), and it is the
% reason "how many neurons do I need?" is not a well-posed question without
% knowing the noise geometry.

% ----- Decoding axis is not a principal component -----
% Finally, connect back to the PCA tutorial. Compute the PCs of the noise
% and ask how the decoding axis relates to them.

% Centre each class separately so that we see the NOISE covariance, not the
% stimulus-driven difference between the two means. (Computing the PCs by
% SVD rather than calling pca() keeps this section toolbox-free and makes
% the connection to the linear algebra tutorial explicit: the right
% singular vectors of the centred data matrix are the eigenvectors of its
% covariance.)
Xc_centered = [XA_c - mean(XA_c,1); XB_c - mean(XB_c,1)];
[~, Sv, coeff_noise] = svd(Xc_centered, 'econ');
lat_noise = diag(Sv).^2 / (size(Xc_centered,1) - 1);

n_show = 20;
overlap = zeros(n_show,1);
for k = 1:n_show
    overlap(k) = abs(coeff_noise(:,k)' * f_prime);
end

subplot(2,3,4); hold on
bar(1:n_show, overlap, 'FaceColor',[0.5 0.4 0.7], 'EdgeColor','none')
xlabel('Noise PC index'); ylabel('|cos angle with signal axis|')
title('The signal is not aligned with the top noise PCs')
grid on

subplot(2,3,5)
plot(1:n_show, 100*lat_noise(1:n_show)/sum(lat_noise), '-o','LineWidth',2,'Color','k')
xlabel('Noise PC index'); ylabel('Variance explained (%)')
title('Noise variance spectrum')
grid on

subplot(2,3,6); hold on
plot(pref_oris, coeff_noise(:,1)/norm(coeff_noise(:,1)), 'LineWidth',1.8, 'Color',[0.5 0.4 0.7])
plot(pref_oris, f_prime, 'k--', 'LineWidth', 2)
plot([0 180],[0 0],':','Color',[0.5 0.5 0.5])
xlabel('Preferred orientation (deg)'); ylabel('Normalized loading')
legend('noise PC1','signal axis','Location','northeast')
title('PC1 of noise vs. the discriminative direction')
xlim([0 180]); set(gca,'XTick',0:30:180)

% The largest source of population variance is a broad, mostly positive
% mode -- a shared gain fluctuation that raises or lowers the whole
% population together. The discriminative direction is the opponent,
% sign-flipping profile we saw in Part III. They are close to orthogonal.
%
% This is the punchline that separates PCA from classification. If you had
% run PCA and kept the top few components, you would have thrown away most
% of the stimulus information, because the stimulus signal here is small in
% variance but large in discriminability. Dimensionality reduction chosen
% by variance is not dimensionality reduction chosen by relevance.

% Homework question 8.
% (a) Derive d'^2 = df' * S^-1 * df for the optimal linear decoder of two
%     Gaussian distributions with common covariance S.
% (b) Explain geometrically why noise along f' cannot be removed by any
%     linear readout, while noise orthogonal to f' can.
% (c) Set c0 to a realistic cortical value (0.1 to 0.15) and re-run. Is the
%     advantage of the optimal decoder still detectable in a single
%     simulation? How many repeats would you need to measure it?
% (d) Set c0 negative (anti-correlated noise between similarly tuned
%     neurons). Does discriminability go up or down? Why do some authors
%     argue that "correlations are harmful" while others argue they can
%     be beneficial?
% (e) Repeat the noise-PC analysis with independent noise. Where does the
%     signal axis sit in the PC ordering then?


%% ========================================================================
%  Part VIII.  Nonlinear classifiers and a head-to-head comparison
%% ========================================================================

% We close by putting every method on the same footing: identical trials,
% identical cross-validation folds, accuracy only on held-out data.
%
% We add one nonlinear method implemented from scratch, k-nearest
% neighbours. It has no training step at all: to classify a new trial,
% find the k most similar training trials in the N-dimensional response
% space and take a majority vote. It makes no assumption about the shape of
% the boundary, which is both its strength and its weakness -- in high
% dimensions with few trials, "nearest" stops being meaningful.

method_names = {'diff-of-means', 'LDA (\lambda=0.05)', 'ridge logistic', ...
                'kNN (k=15)', 'linear SVM'};
acc_cmp = zeros(1, numel(method_names));

% Same folds for every method, so differences reflect the classifiers.
acc_cmp(1) = cv_accuracy(X, y, folds, @(a,b,c) lda_predict(a,b,c,1.0));
acc_cmp(2) = cv_accuracy(X, y, folds, @(a,b,c) lda_predict(a,b,c,0.05));
acc_cmp(3) = cv_accuracy(X, y, folds, @(a,b,c) ...
                logistic_predict(c, logistic_train(a,b,1.0)) > 0.5);
acc_cmp(4) = cv_accuracy(X, y, folds, @(a,b,c) knn_predict(a,b,c,15));
if has_svm
    acc_cmp(5) = cv_accuracy(X, y, folds, @(a,b,c) ...
        double(predict(fitcsvm(a, b, 'KernelFunction','linear', ...
                               'BoxConstraint',1,'Standardize',true), c)));
else
    acc_cmp(5) = cv_accuracy(X, y, folds, @(a,b,c) svm_predict_hinge(a,b,c,1));
end

figure(10); clf
subplot(1,2,1)
bar(acc_cmp, 'FaceColor',[0.35 0.45 0.70], 'EdgeColor','none')
hold on
plot([0 numel(acc_cmp)+1], [0.5 0.5], 'k:', 'LineWidth', 1.5)
set(gca,'XTick', 1:numel(acc_cmp), 'XTickLabel', method_names, 'XTickLabelRotation', 30)
ylabel(sprintf('%d-fold CV accuracy', kfold))
ylim([0.4 1]); grid on
title('All methods, identical folds')

for i = 1:numel(acc_cmp)
    fprintf('%-22s : %.3f\n', method_names{i}, acc_cmp(i));
end

% ----- Where does kNN break down? -----
% In a real recording most neurons are not tuned to the variable you are
% decoding. We simulate that by appending UNINFORMATIVE neurons: units with
% the same mean rate under both stimuli and the same noise, carrying no
% signal at all. A good decoder should ignore them. Watch what each method
% does as they accumulate.

mu_flat = mean((mu_A + mu_B)/2);              % a typical population rate
n_dist_grid = [0 20 50 100 200 400];
lam_show   = [0.05 0.5 0.95];
acc_lin_D  = zeros(numel(lam_show), numel(n_dist_grid));
acc_knn_D  = zeros(1, numel(n_dist_grid));

folds5 = make_folds(numel(y), 5);
for k = 1:numel(n_dist_grid)
    M = n_dist_grid(k);
    if M > 0
        Xd = [X, sample_trials(mu_flat*ones(M,1), numel(y), fano, eye(M))];
    else
        Xd = X;
    end
    for j = 1:numel(lam_show)
        acc_lin_D(j,k) = cv_accuracy(Xd, y, folds5, @(a,b,c) ...
            lda_predict(a, b, c, lam_show(j)));
    end
    acc_knn_D(k) = cv_accuracy(Xd, y, folds5, @(a,b,c) knn_predict(a,b,c,15));
end

subplot(1,2,2); hold on
lam_cols = [0.20 0.40 0.85; 0.15 0.55 0.40; 0.45 0.35 0.70];
for j = 1:numel(lam_show)
    plot(n_dist_grid, acc_lin_D(j,:), '-o','LineWidth',2, ...
         'Color', lam_cols(j,:), 'MarkerFaceColor', lam_cols(j,:))
end
plot(n_dist_grid, acc_knn_D, '-s','LineWidth',2,'Color',[0.85 0.30 0.25], ...
     'MarkerFaceColor',[0.85 0.30 0.25])
plot([0 max(n_dist_grid)], [0.5 0.5], 'k:')
xlabel('Number of uninformative neurons added'); ylabel('CV accuracy')
legend('LDA \lambda=0.05','LDA \lambda=0.5','LDA \lambda=0.95','kNN (k=15)', ...
       'Location','southwest')
title('Curse of dimensionality')
ylim([0.5 0.9]); grid on

fprintf('\nAfter adding %d uninformative neurons:\n', n_dist_grid(end));
for j = 1:numel(lam_show)
    fprintf('  LDA lambda = %.2f : %.3f -> %.3f\n', lam_show(j), ...
            acc_lin_D(j,1), acc_lin_D(j,end));
end
fprintf('  kNN k = 15        : %.3f -> %.3f\n', acc_knn_D(1), acc_knn_D(end));

% Three things are visible at once, and together they are the practical
% summary of this tutorial.
%
% Lightly regularized LDA collapses. With 480 features and 640 training
% trials per fold, lambda = 0.05 is no longer enough regularization, and
% the decoder starts fitting the useless dimensions.
%
% Heavily regularized LDA degrades far more gracefully. The same method
% with lambda = 0.5 or 0.95 loses only a few points, because shrinkage lets
% it assign the useless neurons small weights and carry on. Note also that
% the ordering of the three lambdas REVERSES along the x axis: light
% shrinkage wins with 80 neurons and loses badly with 480. The best lambda
% was therefore not a property
% of the method -- it grew as the problem got harder. Any hyperparameter
% you tune on one data set must be re-tuned when the dimensionality
% changes.
%
% kNN degrades no matter what. It has no weights to shrink: every neuron
% contributes equally to the Euclidean distance, so uninformative
% dimensions inject noise directly into the notion of "nearest". Push this
% far enough and every training trial is roughly equidistant from every
% test trial, and the neighbourhood carries no information at all.
%
% This is why methods that make an explicit assumption about the form of
% the boundary tend to beat assumption-free methods on neural data, where
% the number of recorded units is large and the number of trials is not.
%
% Two further methods you will meet, but which we do not implement here:
%
%   Random forests: ensembles of decision trees, each trained on a
%     bootstrap sample with a random subset of neurons. Robust, handle
%     nonlinearity and feature interactions automatically, and give a
%     natural measure of which neurons matter. In MATLAB: TreeBagger or
%     fitcensemble.
%
%   Neural networks: a stack of linear-plus-nonlinearity layers trained by
%     gradient descent. Overkill for a two-class problem with a few hundred
%     trials, but the natural choice when the decoded variable is
%     continuous, high-dimensional, or unfolds over time.
%
% A closing methodological point. In this simulation the optimal decoder is
% linear by construction, so the elaborate methods cannot beat regularized
% LDA. That is often true of real data too. When a fancy classifier
% outperforms a simple one on neural data, the first question to ask is
% not "what did the network learn?" but "did I cross-validate correctly,
% and is the extra accuracy larger than the spread across folds?"

% Homework question 9.
% (a) Add error bars to the comparison figure by reporting the standard
%     deviation of accuracy ACROSS folds. Are any of the differences
%     between methods larger than the fold-to-fold spread?
% (b) Sweep k in kNN from 1 to 200. Sketch the bias-variance trade-off
%     that the resulting curve reveals. Does the best k change when the
%     uninformative neurons are added?
% (c) Add an RBF-kernel SVM to the comparison. Does it beat the linear
%     methods on this data set? Should it, given how the data were
%     generated?
% (d) Repeat the entire comparison using the correlated responses XA_c and
%     XB_c from Part VII. Which methods suffer most, and does that match
%     your prediction from the geometry?

% Homework question 10 (synthesis).
% You record 150 neurons in V1 while a mouse discriminates 88 from 92
% degrees, and you obtain 60 trials per condition. Design the complete
% analysis: which classifier, what regularization, how you would choose
% it, how you would cross-validate, what null distribution you would use,
% and what control analysis would convince a skeptical reviewer that your
% decoder is reading out orientation rather than running speed, pupil
% size, or slow drift in the recording.


%% ========================================================================
%  Helper functions
%% ========================================================================

function mu = tuning_mean(pref_oris, kappa, stim_ori, baseline, gain)
% Von Mises orientation tuning, 180-degree periodic, peak normalized to 1.
    d  = (stim_ori - pref_oris(:)) * pi/180;
    tc = exp(kappa * cos(2*d)) / exp(kappa);
    mu = baseline + gain * tc;
end

function X = sample_trials(mu, n_trials, fano, L)
% Draw n_trials samples with mean mu and variance fano*mu.
% L is the Cholesky factor (upper triangular) of the desired noise
% correlation matrix; pass eye(N) for independent noise.
    mu = mu(:)';
    Nn = numel(mu);
    z  = randn(n_trials, Nn) * L;              % cov(z) = L'*L
    sd = sqrt(max(fano * mu, eps));
    X  = repmat(mu, n_trials, 1) + z .* repmat(sd, n_trials, 1);
    X  = max(X, 0);                            % rates cannot be negative
end

function C = limited_range_corr(pref_oris, c0, tau_deg)
% Pairwise noise correlation decaying with difference in preferred
% orientation, on a 180-degree circle.
    p  = pref_oris(:);
    d  = abs(p - p');
    d  = min(d, 180 - d);
    C  = c0 * exp(-d / tau_deg);
    C(1:size(C,1)+1:end) = 1;
    C  = (C + C')/2;
end

function A = nearest_pd(C)
% Clip eigenvalues to make a symmetric matrix positive definite.
    [V, D] = eig((C + C')/2);
    d = max(diag(D), 1e-6);
    A = V * diag(d) * V';
    A = (A + A')/2;
end

function [pfa, phit, auc] = roc_curve(r0, r1)
% ROC by explicit criterion sweep. r0 is the noise/class-0 distribution.
    crit = sort(unique([r0(:); r1(:)]), 'descend');
    crit = [crit(1) + 1; crit];
    phit = zeros(numel(crit),1);
    pfa  = zeros(numel(crit),1);
    for i = 1:numel(crit)
        phit(i) = mean(r1 >= crit(i));
        pfa(i)  = mean(r0 >= crit(i));
    end
    auc = trapz(pfa, phit);
end

function auc = mann_whitney_auc(r0, r1)
% AUC as P(r1 > r0) + 0.5*P(r1 == r0), computed from ranks.
    r0 = r0(:); r1 = r1(:);
    n0 = numel(r0); n1 = numel(r1);
    rr = tied_rank([r0; r1]);
    auc = (sum(rr(n0+1:end)) - n1*(n1+1)/2) / (n0*n1);
end

function rk = tied_rank(v)
% Ranks with ties averaged (avoids the Statistics Toolbox dependency).
    [s, ord] = sort(v(:));
    n  = numel(s);
    rk = zeros(n,1);
    i  = 1;
    while i <= n
        j = i;
        while j < n && s(j+1) == s(i)
            j = j + 1;
        end
        rk(i:j) = (i + j)/2;
        i = j + 1;
    end
    out = zeros(n,1);
    out(ord) = rk;
    rk = out;
end

function p = normal_cdf(x)
% Standard normal CDF without the Statistics Toolbox.
    p = 0.5 * (1 + erf(x / sqrt(2)));
end

function S = pooled_cov(X0, X1)
% Within-class covariance pooled across the two classes.
    n0 = size(X0,1); n1 = size(X1,1);
    S  = ((n0-1)*cov(X0) + (n1-1)*cov(X1)) / (n0 + n1 - 2);
end

function [w, b] = lda_train(X0, X1, lambda)
% Fisher linear discriminant with shrinkage toward a scaled identity.
% lambda = 0 is plain LDA; lambda = 1 is the difference-of-means decoder.
    mu0 = mean(X0,1)';
    mu1 = mean(X1,1)';
    S   = pooled_cov(X0, X1);
    if lambda > 0
        S = (1-lambda)*S + lambda * mean(diag(S)) * eye(size(S,1));
    end
    % With fewer trials than neurons S is singular. Rather than let the
    % backslash operator return infinities, fall back to the pseudoinverse,
    % which returns the minimum-norm solution. It still overfits badly --
    % that is the point of the learning-curve figure in Part VI -- but it
    % does so without numerical garbage.
    if rcond(S) < 1e-12 || any(~isfinite(S(:)))
        w = pinv(S) * (mu1 - mu0);
    else
        w = S \ (mu1 - mu0);
    end
    b = -w' * (mu0 + mu1) / 2;
end

function yhat = lda_predict(Xtr, ytr, Xte, lambda)
% Train LDA on (Xtr,ytr) and return predicted labels for Xte.
    [w, b] = lda_train(Xtr(ytr==0,:), Xtr(ytr==1,:), lambda);
    yhat = (Xte*w + b) > 0;
end

function w = logistic_train(X, y, lambda)
% Ridge-penalized logistic regression by iteratively reweighted least
% squares. Returns [intercept; weights]. The intercept is not penalized.
    [n, p] = size(X);
    Xa = [ones(n,1), X];
    w  = zeros(p+1, 1);
    R  = lambda * eye(p+1);  R(1,1) = 0;
    for it = 1:100
        eta = Xa * w;
        mu  = 1 ./ (1 + exp(-eta));
        s   = max(mu .* (1 - mu), 1e-6);
        g   = Xa' * (y - mu) - R * w;
        H   = Xa' * (Xa .* s) + R;
        step = H \ g;
        w = w + step;
        if norm(step) < 1e-8
            break
        end
    end
end

function p = logistic_predict(X, w)
% Predicted P(y = 1) for design matrix X and coefficients [b; w].
    p = 1 ./ (1 + exp(-([ones(size(X,1),1), X] * w)));
end

function [w, b] = hinge_train(X, y, C)
% Soft-margin linear SVM by subgradient descent, as a fallback when
% fitcsvm is unavailable. y is 0/1 and is converted to -1/+1 internally.
    t = 2*y(:) - 1;
    [n, p] = size(X);
    mu = mean(X,1); sd = std(X,0,1); sd(sd==0) = 1;
    Z  = (X - mu) ./ sd;
    w  = zeros(p,1); b = 0;
    for it = 1:3000
        eta = 1 / (1 + 0.01*it);
        m   = t .* (Z*w + b);
        act = m < 1;
        gw  = w - C * (Z(act,:)' * t(act)) / n;
        gb  =    - C * sum(t(act)) / n;
        w = w - eta * gw;
        b = b - eta * gb;
    end
    w = w ./ sd(:);
    b = b - sum(w .* mu(:));
end

function yhat = svm_predict_hinge(Xtr, ytr, Xte, C)
    [w, b] = hinge_train(Xtr, ytr, C);
    yhat = (Xte*w + b) > 0;
end

function yhat = knn_predict(Xtr, ytr, Xte, k)
% k-nearest neighbours with Euclidean distance, no toolbox required.
    d2 = sum(Xte.^2,2) - 2*(Xte*Xtr') + sum(Xtr.^2,2)';
    [~, idx] = sort(d2, 2, 'ascend');
    ytr = ytr(:);
    yhat = mean(ytr(idx(:,1:k)), 2) > 0.5;
end

function fold = make_folds(n, kfold)
% Assign each of n trials to one of kfold groups, balanced in size.
% Generating the folds ONCE and reusing them is what makes a comparison
% between methods a paired comparison rather than a noisy one.
    fold = zeros(n,1);
    fold(randperm(n)) = mod(0:n-1, kfold) + 1;
end

function acc = cv_accuracy(X, y, folds, trainfun)
% k-fold cross-validated balanced accuracy. trainfun has the signature
%   yhat = trainfun(Xtrain, ytrain, Xtest)
% "folds" is either the number of folds (new random folds are drawn) or a
% vector of fold assignments from make_folds (the folds are reused).
    n = numel(y);
    if isscalar(folds)
        fold = make_folds(n, folds);
    else
        fold = folds(:);
    end
    kfold = max(fold);
    correct = zeros(kfold,1);
    for f = 1:kfold
        te = fold == f;
        tr = ~te;
        yhat = trainfun(X(tr,:), y(tr), X(te,:));
        yt = y(te);
        a0 = mean(yhat(yt==0) == 0);
        a1 = mean(yhat(yt==1) == 1);
        correct(f) = (a0 + a1)/2;
    end
    acc = mean(correct);
end

function acc = nested_cv_accuracy(X, y, kfold, lam_grid)
% Nested cross-validation: the inner loop selects lambda, the outer loop
% measures accuracy. This is the honest way to report a tuned classifier.
    n = numel(y);
    fold = zeros(n,1);
    fold(randperm(n)) = mod(0:n-1, kfold) + 1;
    correct = zeros(kfold,1);
    for f = 1:kfold
        te = fold == f;  tr = ~te;
        Xtr = X(tr,:); ytr = y(tr);

        inner = zeros(size(lam_grid));
        for k = 1:numel(lam_grid)
            inner(k) = cv_accuracy(Xtr, ytr, 4, @(a,b,c) ...
                lda_predict(a, b, c, lam_grid(k)));
        end
        [~, kbest] = max(inner);

        yhat = lda_predict(Xtr, ytr, X(te,:), lam_grid(kbest));
        yt = y(te);
        correct(f) = (mean(yhat(yt==0)==0) + mean(yhat(yt==1)==1))/2;
    end
    acc = mean(correct);
end

function d = dprime_along(X0, X1, w)
% d-prime of the projection of two data sets onto a unit vector w.
    p0 = X0*w; p1 = X1*w;
    d  = abs(mean(p1) - mean(p0)) / sqrt((var(p0) + var(p1))/2);
end

function s = numerical_slope(pref_oris, kappa, theta0, gain)
% Derivative of the tuning curve with respect to stimulus orientation,
% evaluated at theta0, for each neuron.
    h  = 0.01;
    m1 = tuning_mean(pref_oris, kappa, theta0 + h, 0, gain);
    m0 = tuning_mean(pref_oris, kappa, theta0 - h, 0, gain);
    s  = (m1 - m0) / (2*h);
end

function [ctr, obs] = calibration_curve(p, y, nbin)
% Observed frequency of class 1 within bins of predicted probability.
    edges = linspace(0, 1, nbin+1);
    ctr = zeros(nbin,1); obs = zeros(nbin,1);
    for b = 1:nbin
        in = p >= edges(b) & p < edges(b+1);
        if sum(in) > 0
            ctr(b) = mean(p(in));
            obs(b) = mean(y(in));
        else
            ctr(b) = NaN; obs(b) = NaN;
        end
    end
end

function thr = interp_threshold(x, auc, target)
% Linearly interpolate the x value at which AUC first reaches target.
    idx = find(auc >= target, 1, 'first');
    if isempty(idx)
        thr = NaN;
    elseif idx == 1
        thr = x(1);
    else
        x0 = x(idx-1); x1 = x(idx);
        a0 = auc(idx-1); a1 = auc(idx);
        thr = x0 + (target - a0) * (x1 - x0) / (a1 - a0);
    end
end

function draw_boundary(w, b, X, col, ~)
% Draw the line w'x + b = 0 within the bounding box of X (2-D only).
    xl = [min(X(:,1)) max(X(:,1))];
    if abs(w(2)) > 1e-12
        yy = -(w(1)*xl + b) / w(2);
        plot(xl, yy, '-', 'Color', col, 'LineWidth', 1.8)
    else
        xv = -b / w(1);
        yl = [min(X(:,2)) max(X(:,2))];
        plot([xv xv], yl, '-', 'Color', col, 'LineWidth', 1.8)
    end
end

function i = find_pref(pref_oris, target)
% Index of the neuron whose preferred orientation is closest to target.
    [~, i] = min(abs(pref_oris - target));
end

function idx = trn_idx_full(n_trials)
% Even-indexed trials from each of the two stimulus blocks, used as a
% training subset for the regularization-path figure.
    idx = [(2:2:n_trials)'; n_trials + (2:2:n_trials)'];
end
