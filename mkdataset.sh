for i in {1..5}
do
	nohup python qutip_dataset_1nn.py --size=6 --dt=0.1   --seed=$i --n_dataset=15000 --file_name=data/periodic/trainset/$i --noise_type=periodic --omega=4  > output_$i\.txt &
done