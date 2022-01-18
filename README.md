# Rain-nowcasting-LC

## 1. Train the model

To train the model use the command :
`$ python3 train_convLSTM.py`

Parameters are defined in `train_convLSTM.py`, checkpoint will be saved in "src/checkpoints" folder at each iterations

## 2. Test the model

To test the model, use the command :
`$ python3 eval_convlstm.py`

Images will be saved in a subfolder Images, you might need to create this folder beforehand.