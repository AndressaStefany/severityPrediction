## Serveurs utilisables

### Cedar
| nœuds | cœurs | mémoire disponible | CPU                                         | stockage     | GPU                                       |
| ----- | ----- | ------------------ | ------------------------------------------- | ------------ | ----------------------------------------- |
| 114   | 24    | 125G ou 128000M    | 2 x Intel E5-2650 v4 Broadwell @ 2.2GHz     | 1 x SSD 800G | 4 x NVIDIA P100 Pascal (mémoire HBM2 12G) |
| 32    | 24    | 250G ou 257000M    | 2 x Intel E5-2650 v4 Broadwell @ 2.2GHz     | 1 x SSD 800G | 4 x NVIDIA P100 Pascal (mémoire HBM2 16G) |
| 192   | 32    | 187G ou 192000M    | 2 x Intel Silver 4216 Cascade Lake @ 2.1GHz | 1 x SSD 480G | 4 x NVIDIA V100 Volta (mémoire HBM2 32G)  |

### Beluga

| nœuds | cœurs           | mémoire disponible | CPU                                   | stockage             | GPU                                                     |
| ----- | --------------- | ------------------ | ------------------------------------- | -------------------- | ------------------------------------------------------- |
| 172   | 40              | 186G ou 191000M    | 2 x Intel Gold 6148 Skylake @ 2.4 GHz | 1 x SSD NVMe de 1.6T | 4 x NVidia V100SXM2 (mémoire 16G), connectés via NVLink |


## Accès

WSL
By hand in terminal
```
# Generate the ssh key
ssh-keygen -t rsa -b 2048
username="your_username_here"
ssh-copy-id $username@beluga.calculcanada.ca
ssh-copy-id $username@cedar.calculcanada.ca
```
In .bashrc
```
username="your_username_here"
alias serverGraham='ssh $username@graham.calculcanada.ca'
alias serverCedar='ssh $username@cedar.calculcanada.ca'
```

Save in file severity

```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --cpus-per-task=1 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --time=00:00:10
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# syntax of 
# Capture the start time
start_time=$(date +"%Y-%m-%d %H:%M:%S")

# loading modules
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch --no-index

python -c "import torch;print(torch.cuda.is_available())"


# Capture the end time
end_time=$(date +"%Y-%m-%d %H:%M:%S")

# Calculate and display the execution time
start_seconds=$(date -d "$start_time" '+%s')
end_seconds=$(date -d "$end_time" '+%s')
execution_time=$((end_seconds - start_seconds))

echo "Execution time: $execution_time seconds"
```

sbatch severity

The result will be in severity-....out and the errors in severity-....err

Check the status of the job with 

```
sq -u ...your username...
```
An alias can be made:

```echo "alias myalias='sq -u rmoine'" >> ~/.bashrc```

## Usage & dossiers


| Name    | Chemin                        | Description / Usage          |
| ------- | ----------------------------- | ---------------------------- |
| HOME    | /home/username/               | 50GB and 500K files per user |
| SCRATCH | /scratch/username/            | 20TB and 1M files per user <br>⚠ Clean periodically |
| PROJECT | /project/def-aloise/username/ | 1TB and 500K files per user. <br> Shared in the research group |
