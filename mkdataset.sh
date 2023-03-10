for i in {1..10}
do
	nohup python qutip_dataset_1nn.py --size=6 --sigma=40 --different_gaussians=100 --seed=$i --n_dataset=15000 --file_name=data/gaussian_driving/trainset_1/$i   > output_$i\.txt &
done