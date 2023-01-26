# probings4mnmt

## Train linear model

```bash
data="bert_english"; python train_linear_weight.py -d ${data} | tee models/train_${data}.log 
```

```bash
optional arguments:
  -h, --help            show this help message and exit
  -d {bert_english,debug}, --data {bert_english,debug}
  --lr LR
  --weight_decay WEIGHT_DECAY
  --batch_size BATCH_SIZE
  --num_updates NUM_UPDATES
  --patience PATIENCE
  --in_dim IN_DIM
  --out_dim OUT_DIM
  --model_type {regressor,classifier}
  --n_trials N_TRIALS   n trials for hyperopt
```

# Debug (fast run)
  
  ```bash
  data="debug"; python train_linear_weight.py -d ${data} | tee models/train_${data}.log
  ```
