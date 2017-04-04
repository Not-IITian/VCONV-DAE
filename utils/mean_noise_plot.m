
slicing_noise = 1 ;
if slicing_noise
slicing_noise_vec = [10,20,30] ;


figure
plot(slicing_noise_vec,mean_ours_slicing,'g-*',slicing_noise_vec,shapenet_mean_slicing,'r-o');
title('Average Error for different amount of slicing noise ')

legend('Ours','Shapenet','Location','northwest')

xlabel('Amount of Slicing Noise') % x-axis label
ylabel('Average Error') % y-axis label

else

noise_vec = [0,30,50] ;
% load mean_vecs_rand.mat

figure
plot(noise_vec,mean_dae,'g-*',noise_vec, mean_shapnet,'r-o', noise_vec, mean_ae, 'b-+');
title('Average Error for different amount of random noise ')

legend('Ours','Shapenet', 'AE', 'Location','northwest')

xlabel('Amount of Random Noise') % x-axis label
ylabel('Average Error') % y-axis label

end