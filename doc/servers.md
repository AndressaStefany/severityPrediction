## Procédure
1. Setup WSL ssh access
2. 
## 1. Choose GPU server

### Cedar
| nœuds | cœurs | mémoire disponible | CPU                                         | stockage     | GPU                                       |
| ----- | ----- | ------------------ | ------------------------------------------- | ------------ | ----------------------------------------- |
| 114   | 24    | 125G ou 128000M    | 2 x Intel E5-2650 v4 Broadwell @ 2.2GHz     | 1 x SSD 800G | 4 x NVIDIA P100 Pascal (mémoire HBM2 12G) |
| 32    | 24    | 250G ou 257000M    | 2 x Intel E5-2650 v4 Broadwell @ 2.2GHz     | 1 x SSD 800G | 4 x NVIDIA P100 Pascal (mémoire HBM2 16G) |
| 192   | 32    | 187G ou 192000M    | 2 x Intel Silver 4216 Cascade Lake @ 2.1GHz | 1 x SSD 480G | 4 x NVIDIA V100 Volta (mémoire HBM2 32G)  |
Max duration of a job: 28 days
### Beluga

| nœuds | cœurs           | mémoire disponible | CPU                                   | stockage             | GPU                                                     |
| ----- | --------------- | ------------------ | ------------------------------------- | -------------------- | ------------------------------------------------------- |
| 172   | 40              | 186G ou 191000M    | 2 x Intel Gold 6148 Skylake @ 2.4 GHz | 1 x SSD NVMe de 1.6T | 4 x NVidia V100SXM2 (mémoire 16G), connectés via NVLink |

Max duration of a job: 168 hours = 7 days
Max 1000 jobs per user
## 2. Setup SSH access

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
alias serverGraham='ssh $username@beluga.calculcanada.ca'
alias serverCedar='ssh $username@cedar.calculcanada.ca'
```
## 3. Adapt your script pathes

You need to adapt your pathes. Here are the folders available:

| Name    | Chemin                        | Description / Usage          |
| ------- | ----------------------------- | ---------------------------- |
| HOME    | /home/username/               | 50GB and 500K files per user |
| SCRATCH | /scratch/username/            | 20TB and 1M files per user <br>⚠ Clean periodically |
| PROJECT | /project/def-aloise/username/ | 1TB and 500K files per user. <br> Shared in the research group |

⚠ For CEDAR move your scripts to SCRATCH or PROJECT
## 4. Copy files

(Optional) add the `severity_launch` file in your repository

```
cd ...to your script directory
scp -r script_folder username@beluga.calculcanada.ca
scp -r script_folder username@cedar.calculcanada.ca
```
severity_launch file

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


# Execution of the script: replace by python path_to_your_script
# Do not forget to change the pathes to absolute pathes
python -c "import torch;print(torch.cuda.is_available())"


# Capture the end time
end_time=$(date +"%Y-%m-%d %H:%M:%S")

# Calculate and display the execution time
start_seconds=$(date -d "$start_time" '+%s')
end_seconds=$(date -d "$end_time" '+%s')
execution_time=$((end_seconds - start_seconds))

echo "Execution time: $execution_time seconds"
```

## 5. Connect & launch & check status

### 5.1 Launch

`sbatch severity_launch`

The result will be in severity_launch-....out and the errors in severity_launch-....err

### 5.2 Check the status of the job 

```
sq -u $USER
```
An alias can be aded to your bashrc with the command *replace the username*:

```echo "alias status='sq -u $USER'" >> ~/.bashrc```

#### 5.2.1 ⚠ Pending job

If your job is running You should see something like
JOBID     USER      ACCOUNT           NAME  ST  TIME_LEFT NODES CPUS TRES_PER_N MIN_MEM  NODELIST (REASON)
41184718   rmoine def-aloise_g launch_test.sh   R       0:41     1    1 gres:gpu:t      8G bg12006 (None)
       
Note the `R` indicating it is running. If it was 



