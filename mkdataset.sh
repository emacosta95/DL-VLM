for i in {1}
do
	nohup python qutip_dataset_1nn.py --size=8 --dt=0.01 --tf=10   --seed=362 --n_dataset=10 --file_name=data/uniform/test_for_adiabatic --noise_type=uniform --rate=0.5 > output_test_$i\.txt &
done