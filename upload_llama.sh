#/bin/bash
server=cedar
root=/scratch
user=rmoine

echo ""
echo "scp -r ./src/llm/llama/*.py rmoine@$server.computecanada.ca:$root/$user/"
# scp -r ./src/llm/llama/*.py rmoine@$server.computecanada.ca:$root/$user/

# echo ""
echo "scp ./src/llm/llama/severity_launch rmoine@$server.computecanada.ca:$root/$user/"
# scp ./src/llm/llama/severity_launch rmoine@$server.computecanada.ca:$root/$user/

# echo 'ssh rmoine@$server.computecanada.ca" "cd $root/$user/&&mkdir data&&mkdir data/predictions"'

# echo ""
echo "scp -r ./data/predictions/out* rmoine@$server.computecanada.ca:$root/$user/data/predictions/"
# scp -r ./data/predictions/out* rmoine@$server.computecanada.ca:$root/$user/data/predictions/

# echo "scp -r ./data/predictions/predictions_v100l.json rmoine@$server.computecanada.ca:$root/$user/data/predictions/"