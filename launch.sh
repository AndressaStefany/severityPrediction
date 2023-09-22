#!/bin/bash
#SBATCH --job-name=severity
#SBATCH --output=severity.out
#SBATCH --error=severity.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
# Capture the start time
start_time=$(date +"%Y-%m-%d %H:%M:%S")

module use /usr/local/src/spack/share/spack/modules/linux-ubuntu20.04-zen2
module load python-3.9.15-gcc-9.4.0-5jyuh3b
module load py-nltk-3.5-gcc-9.4.0-zbkorbu
module load py-numpy-1.24.1-gcc-9.4.0-mtejhor
module load py-optuna/3.2.0-gcc-9.4.0-ba4jncp
module load py-scikit-learn/1.3.1-gcc-9.4.0-wo3twiw

# Start the script
python /home/rmoine/severityPrediction/src/baseline/baseline_functions.py


# Capture the end time
end_time=$(date +"%Y-%m-%d %H:%M:%S")

# Calculate and display the execution time
start_seconds=$(date -d "$start_time" '+%s')
end_seconds=$(date -d "$end_time" '+%s')
execution_time=$((end_seconds - start_seconds))

echo "Execution time: $execution_time seconds"