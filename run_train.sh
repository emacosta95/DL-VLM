time_interval=100




# nohup python train.py --time_interval=$time_interval  --model_type="PixelCNN" --input_channels=1 --output_channels=1 --input_size 8 200 --kernel_size 5 7 --padding 2 3 --pooling=0    --data_path 'data/dataset_h_eff/train_dataset_pixelcnn_ndata_500000_20240520.npz'  --model_name=kohm_sham/cnn_field2field/PixelCNN_field2field_time_steps_100_tf_20_240601_dataset_500k --hidden_channels 80 80 80 80 --keys input_data output_data    --epochs=2000      > output_model_field2field_pixelcnn_1.txt &

nohup python train.py --time_interval=$time_interval  --model_type="LSTM" --input_channels=1 --output_channels=1 --input_size 8 --kernel_size 0 --padding 0 --pooling=0    --data_path 'data/dataset_h_eff/periodic/xxzx_model/dataset_random_rate_random_amplitude_01-08_fixed_initial_state_nbatch_1_batchsize_20000_steps_400_tf_40.0_l_8_240607.npz'  --model_name=kohm_sham/cnn_field2field/LSTM_field2field_random_driving_xxzx_time_steps_100_tf_40_240620_dataset_20k --hidden_channels 200 200 200 200 200 200 200  --keys h h_eff    --epochs=2000      > output_model_field2field_lstm_1.txt &


# nohup python train.py --time_interval=$time_interval  --model_type="REDENTnopooling" --input_channels=1 --output_channels=4 --input_size 8 --kernel_size 5 --padding 2 --pooling=0    --data_path 'data/dataset_h_eff/train_dataset_punctual_periodic_driving_200000.npz'  --model_name=kohm_sham/cnn_density2field/REDENTnopooling_density2field_time_steps_200_tf_20_240602_dataset_200k --hidden_channels 40 40 40 40  --keys density F_density    --epochs=2000      > output_model_density2field_redent_1.txt &

