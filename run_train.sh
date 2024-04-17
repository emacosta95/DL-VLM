time_interval=500




nohup python train.py --time_interval=$time_interval --input_size=8 --model_type="VAE" --input_channels=1 --output_channels=1 --input_size  200 8  --kernel_size 7 5 --padding 3 2 --pooling=2    --data_path 'data/dataset_h_eff/periodic/dataset_periodic_random_rate_0-1_nbatch_1000_batchsize_1000_steps_200_tf_20.0_l_8_240326.npz'  --model_name=kohm_sham/cnn_field2field/VAE_fourier_field2field_time_steps_200_tf_20_240327_dataset_60k --hidden_channels 40 40 40 --keys h h_eff     --epochs=5000      > output_model_field2field_unet_1.txt &


# nohup python train.py --time_interval=$time_interval --model_type="LSTM" --n_layers=5 --input_size=6 --data_path 'data/periodic/train_size_6_tf_10.0_dt_0.1_a_0_1_omega_0_4_n_dataset_150000.npz' --pooling=1 --model_name=lstm_periodic_omega_4_amplitude_1_t_01_l_6_mixed_nt_$time_interval\_150k_pbc --hidden_channels 40       --epochs=3000      > output_tddft_test_lstm_$time_interval\_15k.txt &

#seq2seq_regularization_01_sigma_1_9_c_0_4_set_of_gaussians_100_t_01_l_6_nt_$time_interval\_15k_pbc