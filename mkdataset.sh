for i in {1..10}
do
	nohup python qutip_dataset_1nn.py --size=6 --dt=0.1 --tf=10   --seed=$i --n_dataset=15000 --file_name=data/uniform/trainset/train_$i --noise_type=uniform --c=4 --sigma=9  > output_test_$i\.txt &
done