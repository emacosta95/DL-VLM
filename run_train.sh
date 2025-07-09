time_interval=1001

#nohup python train.py --time_interval=$time_interval  --model_type="TDDFTCNN" --input_channels=1 --output_channels=1 --input_size 8 401 --kernel_size 5 5 --padding 2 2 --pooling=0    --data_path 'data/dataset_h_eff/xxzx_model/dataset_2024-07-14_00-36_CausalCNN.npz'  --model_name=cnn_field2field/CausalCNN_field2field_time_steps_200_tf_20_240819_dataset_50k --hidden_channels 40 40 40 40 40 40 40 40 40 40 40 40 40   --keys h h_eff    --epochs=2000      > output_model_field2field_causalcnn_2.txt &

### INITIALIZE THE TRAIN

nohup python train.py --time_interval=$time_interval  --model_type="LSTM" --input_channels=1 --output_channels=1 --input_size 1 --output_size 1 --kernel_size 5 1 --padding 0 --pooling=0    --data_path 'data/dataset_h_eff/new_analysis_xxzx_model/dataset_2025-06-30_13-44.npz'  --model_name=dataset_2025-06-30_13-44_LSTM_f2f_fixed_initial_state_tf_10_nsteps_1000_250708_dataset_10k --hidden_channels 200 --keys h h_eff    --epochs=2000      > output_model_f2f_lstm_1.txt &

# nohup python train.py --time_interval=$time_interval  --model_type="REDENTnopooling" --input_channels=1 --output_channels=4 --input_size 8 --kernel_size 5 --padding 2 --pooling=0    --data_path 'data/dataset_h_eff/train_dataset_punctual_periodic_driving_200000.npz'  --model_name=kohm_sham/cnn_density2field/REDENTnopooling_density2field_time_steps_200_tf_20_240602_dataset_200k --hidden_channels 40 40 40 40  --keys density F_density    --epochs=2000      > output_model_density2field_redent_1.txt &

### LOADING THE TRAINING

#nohup python train.py --load --name='new_analysis_lstm_field2field/LSTM_field2field_xxzx_nonlinear_auxiliary_field_fixed_initial_state_time_steps_100_tf_10_241118_dataset_50k_[500, 500, 500, 500]_hc_[5, 1]_ks_0_ps_4_nconv_1_nblock' --time_interval=$time_interval  --model_type="LSTM" --input_channels=1 --output_channels=1 --input_size 8 --output_size 8 --kernel_size 5 1 --padding 0 --pooling=0    --data_path 'data/dataset_h_eff/new_analysis_xxzx_model/training_dataset_fixed_initial_state_time_10_nsteps_100_ndata_50000.npz'   --hidden_channels 500 500 500 500   --keys h h_eff    --epochs=2000      > output_model_field2field_lstm_3.txt &
