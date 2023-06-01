for i in {1}
do
	nohup python qutip_dataset_1nn.py --size=8 --dt=0.01 --tf=10   --seed=635 --n_dataset=10 --file_name=data/disorder/test_for_adiabatic --noise_type=disorder --rate=2.  > output_test_$i\.txt &
done