# RL Final Project-REDQ source code

This contains the source code implemented for the REDQ algorithm. Paper linke: https://arxiv.org/abs/2101.05982.

## How to use it

You can simply run the next command to get it and run.

```bash
git clone https://github.com/sumuzhe317/RL-REDQ.git
cd RL-REDQ

python3 -m venv RL-REDQ
python3 -m pip install -r requirements.txt

python3 main_train.py --debug
```

Note that if you want to know the hyperparameters about the train, you can run the next command:

```bash
python3 main_train.py -h
```

Then you can get that:

```bash
usage: main_train.py [-h] [--env ENV] [--seed SEED] [--epochs EPOCHS] [--exp_name EXP_NAME]                                                                          
                     [--data_dir DATA_DIR] [--debug] [--cuda_device [CUDA_DEVICE]] [--lr [LR]]                                                                       
                     [--replay_size [REPLAY_SIZE]] [--batch_size [BATCH_SIZE]] [--gamma [GAMMA]]                                                                     
                     [--polyak [POLYAK]] [--alpha [ALPHA]] [--start_steps [START_STEPS]]                                                                             
                     [--utd_ratio [UTD_RATIO]] [--num_Q [NUM_Q]] [--num_min [NUM_MIN]]                                                                               
                     [--policy_update_delay [POLICY_UPDATE_DELAY]]                                                                                                   

The python code to train a agent using redq                                       

optional arguments:                      
  -h, --help            show this help message and exit                                                                                                              
  --env ENV             The env to train the agent                                                                                                                   
  --seed SEED, -s SEED  The seed to initialize some parameters                                                                                                       
  --epochs EPOCHS       The epochs to train                                       
  --exp_name EXP_NAME   The exp name used to store the train data                                                                                                    
  --data_dir DATA_DIR   The directory to store the data                                                                                                              
  --debug               The quick check for the code                                                                                                                 
  --cuda_device [CUDA_DEVICE]                                                     
                        Which GPU will be used                                    
  --lr [LR]             The learning rate                                         
  --replay_size [REPLAY_SIZE]                                                     
                        The replaybuffer\'s size                                   
  --batch_size [BATCH_SIZE]                                                       
                        The batch size                                            
  --gamma [GAMMA]       The discount factor                                       
  --polyak [POLYAK]     Hyperparameter for polyak averaged target networks    
  --alpha [ALPHA]       SAC entropy hyperparameter
  --start_steps [START_STEPS]
                        The number of random data collected in the beginning of training
  --utd_ratio [UTD_RATIO]
                        The update-to-data ratio
  --num_Q [NUM_Q]       The number of Q networks in the Q ensemble
  --num_min [NUM_MIN]   The number of sampled Q values to take minimal from
  --policy_update_delay [POLICY_UPDATE_DELAY]
                        how many updates until we update policy network
```

You can run the below code to train the agent:

```bash
python3 main_train.py
```