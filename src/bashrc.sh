# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# User specific aliases and functions
# Aliases to jump to folder
alias s='cd /scratch/$USER/';
alias p='cd /project/def-aloise/$USER/';
# Aliases to monitor jobs
squeue_command="squeue -u $USER --format=\"%.18i %.40j %.8u %.2t %.10M %.4D %.5C %b %.10N %.6m %R\"";
alias st='watch -n5 $squeue_command';
alias status="squeue -u $USER --format=\"%.18i %.40j %.8u %.2t %.10M %.4D %.5C %b %.10N %.6m %R\"";
alias sb='sbatch';
detail () {
        scontrol show job $1;
}
# Other aliases
catL () {
        cat $(ls -1 | grep $1 | tail -1);
};
alias c='clear&&cd .&&pwd';
alias bashrc='vim ~/.bashrc';
alias projets='cd /mnt/c/users/robin/documents/projets/';
alias severity='cd /mnt/c/Users/robin/Documents/projets/severityPrediction/';
alias endsound='paplay /usr/share/sounds/freedesktop/stereo/complete.oga';
# To display the current path each time
shopt -s autocd;
shopt -s expand_aliases;
PROMPT_COMMAND='ls;';
PS1="\[\e[34m\]"\\u@\\h:"\[\e[32m\]"\$PWD\\n\\d" "\\t\\n"\[\e[0m\]"'> ';