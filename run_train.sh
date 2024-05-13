time_interval=200




nohup python train.py --time_interval=$time_interval  --model_type="LSTM" --input_channels=1 --output_channels=1 --input_size 8  --kernel_size 0 --padding 0 --pooling=0    --data_path 'data/dataset_h_eff/periodic/dataset_periodic_random_rate_03-1_random_amplitude_01-08_fixed_initial_state_nbatch_1_batchsize_100000_steps_400_tf_40.0_l_8_240509.npz'  --model_name=kohm_sham/cnn_field2field/LSTM_field2field_time_steps_200_tf_20_240510_dataset_20k --hidden_channels 200 200 200 200 200 200 200 200 200 200 --keys h h_eff    --epochs=2000      > output_model_field2field_lstm_2.txt &


# nohup python train.py --time_interval=$time_interval --model_type="LSTM" --n_layers=5 --input_size=6 --data_path 'data/periodic/train_size_6_tf_10.0_dt_0.1_a_0_1_omega_0_4_n_dataset_150000.npz' --pooling=1 --model_name=lstm_periodic_omega_4_amplitude_1_t_01_l_6_mixed_nt_$time_interval\_150k_pbc --hidden_channels 40       --epochs=3000      > output_tddft_test_lstm_$time_interval\_15k.txt &

#seq2seq_regularization_01_sigma_1_9_c_0_4_set_of_gaussians_100_t_01_l_6_nt_$time_interval\_15k_pbc