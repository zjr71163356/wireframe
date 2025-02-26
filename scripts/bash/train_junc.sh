#chmod +x train_junc.sh 
cd ../../junc
python3 main.py --exp 1 --json --gpu 0 --balance
