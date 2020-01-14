# Domain Adaptation : Maximum Classifier Discrepancy

## Prerequisite
### create virtual environment
```
cd ~
virtualenv MCD
echo "export \
PYTHONPATH="$PYTHONPATH:\
<path_to_repo>/ML2019FALL/final/src/model:\
<path_to_repo>/ML2019FALL/final/src/datasets:\
<path_to_repo>/ML2019FALL/final/src/utils" >> MCD/bin/activate
source MCD/bin/activate
```
### install packages
```
cd ~
git clone git@github.com:BennyTW/ML2019FALL.git
cd ~/ML2019FALL/final/
pip install -r requirements.txt
```


## Usage
```
python main.py -h
```

### train model and log std output
```
nohup python main.py --num_k 5 --save_model --resume_epoch 180 &> training.log&
```


### test model 
```
python main.py --num_k 5 --resume_epoch 180 --eval_only
```
