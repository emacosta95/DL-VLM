for i in {1}
do
	nohup python qutip_dataset_1nn.py --size=6 --dt=0.1 --tf=20   --seed=326 --n_dataset=200 --file_name=data/periodic/test --noise_type=periodic --omega=4  > output_test.txt &
done