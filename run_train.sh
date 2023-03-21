time_interval=96




nohup python train.py --time_interval=$time_interval --model_type="TDDFTCNN" --input_channels=1 --kernel_size 22 3 --data_path 'data/periodic/train_size_6_tf_10.0_dt_0.1_a_0_1_omega_0_4_n_dataset_15000.npz' --pooling=1 --model_name=causalcnn_periodic_omega_4_amplitude_1_t_01_l_6_mixed_nt_$time_interval\_15k_pbc --hidden_channels 40 40 40 40      --epochs=3000      > output_tddft_test_$time_interval\_superkernel_15k.txt &


# nohup python train.py --time_interval=$time_interval --model_type="LSTM" --n_layers=5 --input_size=6 --data_path 'data/periodic/train_size_6_tf_10.0_dt_0.1_a_0_1_omega_0_4_n_dataset_150000.npz' --pooling=1 --model_name=lstm_periodic_omega_4_amplitude_1_t_01_l_6_mixed_nt_$time_interval\_150k_pbc --hidden_channels 40       --epochs=3000      > output_tddft_test_lstm_$time_interval\_15k.txt &