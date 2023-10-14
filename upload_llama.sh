#/bin/bash
server=cedar
root=/project/def-aloise
user=rmoine

echo ""
echo "scp -r ./src/llm/llama/*.py rmoine@$server.computecanada.ca:$root/$user/"

# echo ""
echo "scp ./src/llm/llama/severity_launch rmoine@$server.computecanada.ca:$root/$user/"

echo 'ssh rmoine@$server.computecanada.ca" "cd $root/$user/&&mkdir data&&mkdir data/"'

# echo ""
echo "scp -r ./data/llm/data_preprocessed_tokens_v2.json rmoine@$server.computecanada.ca:$root/$user/data/"
