#/bin/bash
server=graham
root_path=/project/def-aloise/rmoine/data
addr=rmoine@$server.computecanada.ca:
# rsync -avz --info=progress2 $addr$root_path/qlora_finetune_* .
rsync -rltv --info=progress2 $addr$root_path/qlora_finetune_*redo .