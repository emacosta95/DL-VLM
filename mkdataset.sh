for i in {1}
do
	nohup python qutip_dataset_1nn.py --size=8 --dt=0.001 --tf=10 --h0=5. --hf=0.   --seed=6985 --n_dataset=50 --file_name=data/uniform/test_for_adiabatic --noise_type=uniform --rate=3 > output_test_$i\.txt &
done