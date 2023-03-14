time_interval=64
sigma=40
n_dataset=15000


nohup python train.py --time_interval=$time_interval --model_type=CausalUnet --kernel_size 5 3 --data_path "data/gaussian_driving/simulation_size_6_tf_10.0_dt_0.05_sigma_10_20_c_0_4.0_noise_100_n_dataset_15000.npz" --model_name=unet_gaussian_driving_sigma_10_20_t_005_l_6_nt_$time_interval\_15k_pbc --hidden_channels 40 80 160 320   --padding_mode='zeros'  --epochs=3000 --input_channels=1   > output_tddft_test_2.txt &