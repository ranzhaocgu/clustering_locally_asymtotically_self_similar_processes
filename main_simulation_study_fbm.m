% clear; clc;
% rng(2022)

% fBm vs. mBm
n_clusters = 2;
obs_num_per_cluster = 25;
obs_num_clusters = obs_num_per_cluster * n_clusters;

total_time_steps = 100;
obs_num_per_step = 3;
initial_points = 5;
total_num_observations = total_time_steps * obs_num_per_step + initial_points;
total_num_paths = 100;
cluster_ind = ones(obs_num_clusters, 1);
cluster_ind((obs_num_clusters/2):obs_num_clusters) = 2;

obs_chain = zeros(total_num_observations, obs_num_clusters, total_num_paths);
Hfunc = @(t) 0.8 + 0.1 * sin(pi * t);
t = linspace(0,1,total_num_observations);
for i = 1:total_num_paths
    for z = 1:obs_num_clusters
        if cluster_ind(z) == 1
            obs_chain(2:end, z, i) = diff(mbmlevinson(total_num_observations, Hfunc(t)), 1);
        else
            obs_chain(2:end, z, i) = diff(fbmlevinson(total_num_observations,0.8), 1);
        end
    end
end

%% offline setting
test_time_steps = 100; 
test_num_sims = 100; 
miscls_rate_offline_algo1 = zeros(test_time_steps, test_num_sims);
miscls_rate_offline_algo2 = zeros(test_time_steps, test_num_sims);
avg_miscls_rate_offline_algo1 = zeros(test_time_steps,1);
avg_miscls_rate_offline_algo2 = zeros(test_time_steps,1);
initial_points = 5;

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

end

% plot of clsutering results
% x = 1:test_time_steps;
% figure
% plot(x, avg_miscls_rate_offline_algo1(1:test_time_steps), 'b', 'LineWidth', 2)
% hold on
% plot(x, avg_miscls_rate_offline_algo2(1:test_time_steps), '-.r', 'LineWidth', 2)
% hold off
% title('Offline Dataset with Covariance Distance Clustering')
% xlabel('time step')
% ylabel('misclassification rate')
% legend('Algorithm 1', 'Algorithm 2')       

save('mpdi1.mat')