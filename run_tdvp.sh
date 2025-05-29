#!/bin/bash

# Number of parallel runs
nruns=10

# Delay (in seconds) between starting each job
delay=10

# Julia script to run (update if named differently)
script="DatasetWithTDVP.jl"

for ((i=0; i<nruns; i++)); do
  echo "Launching run $((i+1))..."
  
  # Run Julia script in background
  nohup julia $script > output_julia_$i.txt &

  # Wait before next launch
  sleep $delay


done

# Wait for all background jobs to finish
wait

echo "All $nruns runs completed."