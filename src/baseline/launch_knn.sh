#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --time=6-00:00:00           # duration (JJ-HH:MM:SS)
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out
#SBATCH --mail-user=robin.moine456@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem=64G # Request 8 GB of RAM

# syntax of
# Capture the start time
start_time=$(date +"%Y-%m-%d %H:%M:%S")

# loading modules
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install numpy pandas optuna scikit-learn nltk --no-index

# Execution of the script: replace by python path_to_your_script
# Do not forget to change the pathes to absolute pathes (dont hesitate to use $USER variable)
python /scratch/$USER/severityPrediction/src/baseline/baseline_functions.py KNN


# Capture the end time
end_time=$(date +"%Y-%m-%d %H:%M:%S")

# Calculate and display the execution time
start_seconds=$(date -d "$start_time" '+%s')
end_seconds=$(date -d "$end_time" '+%s')
execution_time=$((end_seconds - start_seconds))

echo "Execution time: $execution_time seconds"