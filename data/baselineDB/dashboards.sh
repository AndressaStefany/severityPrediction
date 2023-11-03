
cd /mnt/c/Users/robin/Documents/projets/severityPrediction/data/baselineDB
conda activate severityPrediction;
optuna-dashboard sqlite:///study-svc.db --port 8081

cd /mnt/c/Users/robin/Documents/projets/severityPrediction/data/baselineDB
conda activate severityPrediction;
optuna-dashboard sqlite:///study-knn.db --port 8082

cd /mnt/c/Users/robin/Documents/projets/severityPrediction/data/baselineDB
conda activate severityPrediction;
optuna-dashboard sqlite:///study-bayesian-networks.db --port 8083