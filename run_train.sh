time_interval=72




nohup python train.py --time_interval=$time_interval --input_size=6 --model_type="UnetLSTM" --input_channels=1 --kernel_size 3 --data_path 'data/periodic/train_size_6_tf_10.0_dt_0.1_a_0_1_omega_0_4_n_dataset_15000.npz' --pooling=1 --model_name=unetlstm/periodic/test_4 --hidden_channels 40 40 40 40   --latent_dimension=6 --hidden_neurons=100 --lstm_layers=5 --n_layers=1     --epochs=3000   --regularization=2   > output_test_4_cnnlstm.txt &


# nohup python train.py --time_interval=$time_interval --model_type="LSTM" --n_layers=5 --input_size=6 --data_path 'data/periodic/train_size_6_tf_10.0_dt_0.1_a_0_1_omega_0_4_n_dataset_150000.npz' --pooling=1 --model_name=lstm_periodic_omega_4_amplitude_1_t_01_l_6_mixed_nt_$time_interval\_150k_pbc --hidden_channels 40       --epochs=3000      > output_tddft_test_lstm_$time_interval\_15k.txt &

#seq2seq_regularization_01_sigma_1_9_c_0_4_set_of_gaussians_100_t_01_l_6_nt_$time_interval\_15k_pbc