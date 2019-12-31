% %% Generate weakly stationary paths
% clear; clc;
% rng(2018)
% % define simulation parameters
% n_clusters = 5;
% obs_num_per_cluster = 10;
% H = -0.4:0.2:0.4;
% total_time_steps = 100;
% obs_num_per_step = 3;
% total_num_observations = total_time_steps * obs_num_per_step;
% total_num_paths = 100;
% 
% % simulation weakly stationary stochastic processes
% [obs_chain, cluster_ind] = sim_wssp_paths(n_clusters, obs_num_per_cluster, H, ...
%                          total_num_observations, total_num_paths);
% save('mBm_sin.mat')

%% Generate weakly stationary paths using sin/cos function
% clear; clc;
% rng(2018)
% define simulation parameters
n_clusters = 5;
obs_num_per_cluster = 10;
total_time_steps = 100;
obs_num_per_step = 3;
initial_points = 5;
total_num_observations = total_time_steps * obs_num_per_step + initial_points;
total_num_paths = 100;

% simulation weakly stationary stochastic processes
obs_num_clusters = obs_num_per_cluster * n_clusters;
obs_chain = zeros(total_num_observations, obs_num_clusters, total_num_paths);
cluster_ind = (floor((0:49)./ 10) + 1)';
t = linspace(0,1,total_num_observations);

for i = 1:total_num_paths
    for z = 1:obs_num_clusters
        if cluster_ind(z) == 1
            % Hfunc = @(t) 0.7 + 0.2 * sin(4 * pi * t + pi/2);
            Hfunc = @(t) 0.5 + 0.4 * sin(pi * t);
            obs_chain(2:end, z, i) = diff(mbmlevinson(total_num_observations, Hfunc(t)), 1);
        elseif cluster_ind(z) == 2
            % Hfunc = @(t) 0.3 + 0.2 * sin(4 * pi * t + 2 * pi/2);
            Hfunc = @(t) 0.5 + 0.2 * sin(pi * t);
            obs_chain(2:end, z, i) = diff(mbmlevinson(total_num_observations, Hfunc(t)), 1);
        elseif cluster_ind(z) == 3
            % Hfunc = @(t) 0.7 + 0.2 * sin(4 * pi * t + 3 * pi/2);
            Hfunc = @(t) 0.5 + 0.0 * sin(pi * t);
            obs_chain(2:end, z, i) = diff(mbmlevinson(total_num_observations, Hfunc(t)), 1);
        elseif cluster_ind(z) == 4
            % Hfunc = @(t) 0.3 + 0.2 * sin(4 * pi * t + 4 * pi/2);
            Hfunc = @(t) 0.5 - 0.2 * sin(pi * t);
            obs_chain(2:end, z, i) = diff(mbmlevinson(total_num_observations, Hfunc(t)), 1);
        elseif cluster_ind(z) == 5
            % Hfunc = @(t) 0.5 + 0 * sin(4 * pi * t + 5 * pi/2);
            Hfunc = @(t) 0.5 - 0.4 * sin(pi * t);
            obs_chain(2:end, z, i) = diff(mbmlevinson(total_num_observations, Hfunc(t)), 1);
        end
    end
end

% save('mBm_sin.mat')
                     
%% Offline dataset experiments
% load('mBm.mat')
test_time_steps = 30; 
test_num_sims = 100; 
miscls_rate_offline_algo1 = zeros(test_time_steps, test_num_sims);
miscls_rate_offline_algo2 = zeros(test_time_steps, test_num_sims);
avg_miscls_rate_offline_algo1 = zeros(test_time_steps,1);
avg_miscls_rate_offline_algo2 = zeros(test_time_steps,1);

for k = 1:test_time_steps
    parfor sim = 1:test_num_sims
    % parfor sim = 1:test_num_sims;  % parallel computing if necessary
        
        % scale the obsersed times series to be mean 0
        obs = obs_chain(1:(initial_points + k * obs_num_per_step), :, sim)';
        obs = scale_mean(obs, 0);
        
        % full matrix is observed under offline dataset
        obs_idx = ones(size(obs));
        
        % clustering the observed time series
        [I_chain_algo1, dm] = unsup_wssp_offline_algo(obs, obs_idx, n_clusters);
        [I_chain_algo2, ~] = unsup_wssp_online_algo(obs, obs_idx, n_clusters, dm);
       
        % calculate misclassification rate
        miscls_rate_offline_algo1(k,sim) = misclassify_rate(I_chain_algo1, cluster_ind);
        miscls_rate_offline_algo2(k,sim) = misclassify_rate(I_chain_algo2, cluster_ind);

        fprintf('Offline simluation iter %i for time step %i. \n', sim, k)
    end
    avg_miscls_rate_offline_algo1(k) = mean(miscls_rate_offline_algo1(k,:));
    avg_miscls_rate_offline_algo2(k) = mean(miscls_rate_offline_algo2(k,:));
    
    if mod(k,5) == 0
        save('sim_results_test.mat')
    end
end

% plot of clsutering results
x = 1:test_time_steps;
figure
plot(x, avg_miscls_rate_offline_algo1(1:test_time_steps), 'b', 'LineWidth', 2)
hold on
plot(x, avg_miscls_rate_offline_algo2(1:test_time_steps), '-.r', 'LineWidth', 2)
hold off
title('Offline Dataset with Covariance Distance Clustering')
xlabel('time step')
ylabel('misclassification rate')
legend('Algorithm 1', 'Algorithm 2')        
save('sim_results_test.mat')


%% Online dataset experiments
test_num_sims = 100; 
miscls_rate_online_algo1 = zeros(test_time_steps, test_num_sims);
miscls_rate_online_algo2 = zeros(test_time_steps, test_num_sims);
avg_miscls_rate_online_algo1 = zeros(test_time_steps,1);
avg_miscls_rate_online_algo2 = zeros(test_time_steps,1);

for k = 1:test_time_steps
    parfor sim = 1:test_num_sims
    % parfor sim = 1:test_num_sims;  % parallel computing if necessary
        
        % scale the obsersed times series to be mean 0
        obs = obs_chain(1:(initial_points + k * obs_num_per_step), :, sim)';
                     
        % full matrix is observed under offline dataset
        obs_idx = zeros(size(obs));
        for i = 1:obs_num_per_cluster:(n_clusters * obs_num_per_cluster)
            obs_idx(i:(i+4), :) = 1;
            for j = 1:(obs_num_per_cluster - 5)
                if k > j * 10
                    obs_idx(5+j, (j * 10 * obs_num_per_step + 1):end) = 1;
                end
            end
        end
        
        keep_idx = find(sum(obs_idx,2) ~= 0);
        obs = obs(keep_idx, :);
        obs_idx = obs_idx(keep_idx, :);
        cluster_ind_online = cluster_ind(keep_idx);
        
        % clustering the observed time series
        [I_chain_algo1, dm] = unsup_wssp_offline_algo(obs, obs_idx, n_clusters);
        [I_chain_algo2, ~] = unsup_wssp_online_algo(obs, obs_idx, n_clusters, dm);
       
        % calculate misclassification rate
        miscls_rate_online_algo1(k,sim) = misclassify_rate(I_chain_algo1, cluster_ind_online);
        miscls_rate_online_algo2(k,sim) = misclassify_rate(I_chain_algo2, cluster_ind_online);

        fprintf('Online simluation iter %i for time step %i. \n', sim, k)
    end
    avg_miscls_rate_online_algo1(k) = mean(miscls_rate_online_algo1(k,:));
    avg_miscls_rate_online_algo2(k) = mean(miscls_rate_online_algo2(k,:));
    
    if mod(k,5) == 0
        save('sim_results_test.mat')
    end
end

% plot of clsutering results
x = 1:test_time_steps;
figure
plot(x, avg_miscls_rate_online_algo1, 'b', 'LineWidth', 2)
hold on
plot(x, avg_miscls_rate_online_algo2, '-.r', 'LineWidth', 2)
hold off
title('Online Dataset with Covariance Distance Clustering')
xlabel('time step')
ylabel('misclassification rate')
legend('Algorithm 1', 'Algorithm 2')     
save('sim_results_test.mat')