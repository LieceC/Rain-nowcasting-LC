# Rain-nowcasting-LC

All of the models are done for a training on a meteonet-dataset

Be careful, you need to change the folder path of the data in `train_convLSTM.py` and `train_convLSTM_with_wind.py`.
We also deleted from the dataset all PPMatrix folders since we don't use them to build the dataset.
You may get an error if the dataset still contains those folders.

## 1.1 Train the model without wind

To train the model without wind use the command :
`$ python3 train_convLSTM.py --epochs 205 --batch_size 2 --length 12 --hidden_dim 64 --kernel_size 3`

We can change the learning rate et weight decay of the network in the file `train_convLSTM.py` line 16 and 17.

## 1.2 Train the model with wind
To train the model with wind use the command :
`$ python3 train_convLSTM_with_wind.py`

Parameters are defined in `train_convLSTM_with_wind.py`, checkpoints will be saved in "src/checkpoints" folder at each iteration.
You can change parameters at line 14 and 15 for the learning rate and weight decay, for the batch size you can set it
in the call function located in the main.

The models saved their differents metrics after some steps in a runs folder, letting use analyse the evolution of our
training through tensorboard.

## 2. Test the model

To test without wind, use the command :
`$ python3 eval_convlstm.py`

To test with wind, use the command :
`$ python3 eval_convlstm_wind.py`

You might want to change the checkpoints to yours, this can be done by changing the path at the start of eval function,
in the torch.load functions.

Images will be saved in a subfolder Images, you might need to create this folder beforehand.
