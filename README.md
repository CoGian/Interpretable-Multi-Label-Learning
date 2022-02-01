# Interpretable-Multi-Label-Learning

## How to run run_experiments_server.sh
It will automatically download the datasets and trained models, it will install all needed libraries, and will start running the experiments. 
```bash
nohup bash run_experiments_server.sh &
```
you can see the progress with:
```bash
tail -f nohup.out 
```
It will finish when it outputs "finished"
