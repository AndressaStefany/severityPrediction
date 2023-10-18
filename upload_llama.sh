#/bin/bash
server=cedar
root=/project/def-aloise
user=rmoine
root_project=/mnt/c/Users/robin/Documents/projets/severityPrediction

echo ""
echo "scp -r $root_project/src/llm/llama/*.py rmoine@$server.computecanada.ca:$root/$user/"

# echo ""
echo "scp $root_project/src/llm/llama/severity_launch rmoine@$server.computecanada.ca:$root/$user/"
echo "scp $root_project/src/llm/llama/inference_* rmoine@$server.computecanada.ca:$root/$user/"
echo "scp $root_project/src/llm/llama/embeddings_gen_* rmoine@$server.computecanada.ca:$root/$user/"
echo "scp $root_project/src/llm/llama/launch_* rmoine@$server.computecanada.ca:$root/$user/"
echo "scp $root_project/src/llm/llama/finetune* rmoine@$server.computecanada.ca:$root/$user/"

echo 'ssh rmoine@$server.computecanada.ca" "cd $root/$user/&&mkdir data&&mkdir data/"'

# echo ""
echo "scp -r $root_project/data/llm/data_preprocessed_tokens_v3.json rmoine@$server.computecanada.ca:$root/$user/data/"

echo "scp -r $root_project/data/llm/data_preprocessed_tokens_v2.json rmoine@$server.computecanada.ca:$root/$user/data/"
