time_interval=500




nohup python train.py --time_interval=$time_interval --input_size=8 --model_type="REDENTnopooling2D" --input_channels=2 --output_channels=2 --input_size=8 --kernel_size 15 5  --padding 7 2  --data_path 'data/dataset_h_eff/periodic/fourier_transform/dataset_periodic_fourier_random_rate_JUSTEFFECTIVEFIELD_nbatch_100_batchsize_1000_steps_200_tf_20.0_l_8_240227.npz' --pooling=1 --model_name=kohm_sham/cnn_field2field/REDENT2D_fourier_field2field_periodic_random_rate_JUSTEFFECTIVEFIELD_time_steps_200_tf_20_240228_periodic_dataset --hidden_channels 80 80 80 80  --keys h potential     --epochs=1000      > output_model_field2field_cnntddft_2.txt &


# nohup python train.py --time_interval=$time_interval --model_type="LSTM" --n_layers=5 --input_size=6 --data_path 'data/periodic/train_size_6_tf_10.0_dt_0.1_a_0_1_omega_0_4_n_dataset_150000.npz' --pooling=1 --model_name=lstm_periodic_omega_4_amplitude_1_t_01_l_6_mixed_nt_$time_interval\_150k_pbc --hidden_channels 40       --epochs=3000      > output_tddft_test_lstm_$time_interval\_15k.txt &

#seq2seq_regularization_01_sigma_1_9_c_0_4_set_of_gaussians_100_t_01_l_6_nt_$time_interval\_15k_pbc