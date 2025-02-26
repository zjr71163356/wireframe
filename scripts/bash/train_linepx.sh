#chmod +x train_linepx.sh
cd linepx
python3 main.py --netType stackedHGB --GPUs 0 --LR 0.001 --batchSize 4
