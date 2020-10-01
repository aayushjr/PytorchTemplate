from torch import cuda

# Set all config parameters here

n_epochs = 100
batch_size = 20
learning_rate = 1e-3
weight_decay = 1e-7
step_size = 60
save = False
save_dir='./data/models'

#optimizer_type = 'SGD'
optimizer_type = 'Adam'

resume = False
checkpoint = 'model_28_99.34.pth'