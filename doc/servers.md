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
## 4. Copy files in the correct directory (see above)

(Optional) add the `severity_launch` file in your repository

```
cd ...to your script directory
scp -r script_folder username@beluga.calculcanada.ca:/project/def-aloise/...
scp -r script_folder username@cedar.calculcanada.ca:/project/def-aloise/...
```
severity_launch file:
to fill first:
- email address (or delete the two last_lines)
- how many CPUs (for efficient data loading maybe or leave it at 1)
- absolute script path (see chapter 3)

```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1 # on cedar choose one v100l gpu but can be also p100l
#SBATCH --cpus-per-task=1 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --time=00:10:00           # duration (JJ-HH:MM:SS)
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out
#SBATCH --mail-user=youemail
#SBATCH --mail-type=ALL
#SBATCH --mem=8G # Request 8 GB of RAM

# syntax of 
# Capture the start time
start_time=$(date +"%Y-%m-%d %H:%M:%S")

# loading modules
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch  --no-index
pip install transformers accelerate huggingface_hub xformers --no-index

# Execution of the script: replace by python path_to_your_script
# Do not forget to change the pathes to absolute pathes (dont hesitate to use $USER variable)
python /project/def-aloise/$USER/main.py


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
       
Note the `R` indicating it is running. If it was `PD` it means that it can not find ressources for the job or that you do not have enough priority. Try to reduce the memory requirement, number of cpus if possible. **Do not remove the min_memory requirement of 8GB because otherwise it stays pending**

### 5.2.2 Stopping a task

If mistake, to not use too computing time (limited to group)

scancel JOBID

JOBID that you get with squeue (sq -u $USER)
## Annex: slurm status

Jobs typically pass through several states in the course of their execution. The typical states are PENDING, RUNNING, SUSPENDED, COMPLETING, and COMPLETED. An explanation of each state follows.
BF BOOT_FAIL
Job terminated due to launch failure, typically due to a hardware failure (e.g. unable to boot the node or block and the job can not be requeued).
CA CANCELLED
Job was explicitly cancelled by the user or system administrator. The job may or may not have been initiated.
CD COMPLETED
Job has terminated all processes on all nodes with an exit code of zero.
CF CONFIGURING
Job has been allocated resources, but are waiting for them to become ready for use (e.g. booting).
CG COMPLETING
Job is in the process of completing. Some processes on some nodes may still be active.
DL DEADLINE
Job terminated on deadline.
F FAILED
Job terminated with non-zero exit code or other failure condition.
NF NODE_FAIL
Job terminated due to failure of one or more allocated nodes.
OOM OUT_OF_MEMORY
Job experienced out of memory error.
PD PENDING
Job is awaiting resource allocation.
PR PREEMPTED
Job terminated due to preemption.
R RUNNING
Job currently has an allocation.
RD RESV_DEL_HOLD
Job is being held after requested reservation was deleted.
RF REQUEUE_FED
Job is being requeued by a federation.
RH REQUEUE_HOLD
Held job is being requeued.
RQ REQUEUED
Completing job is being requeued.
RS RESIZING
Job is about to change size.
RV REVOKED
Sibling was removed from cluster due to other cluster starting the job.
SI SIGNALING
Job is being signaled.
SE SPECIAL_EXIT
The job was requeued in a special state. This state can be set by users, typically in EpilogSlurmctld, if the job has terminated with a particular exit value.
SO STAGE_OUT
Job is staging out files.
ST STOPPED
Job has an allocation, but execution has been stopped with SIGSTOP signal. CPUS have been retained by this job.
S SUSPENDED
Job has an allocation, but execution has been suspended and CPUs have been released for other jobs.
TO TIMEOUT
Job terminated upon reaching its time limit.





