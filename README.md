# Convert training data from other data types to tensor
## Numpy array
`X=torch.from_numpy(X).type(torch.float)`
## int
`y=torch.tensor(y)`

# Put model, training and validation dataset (tensors) to GPU device
```
device = 'cuda' if torch.cuda.is_available else 'cpu'
model=model_class.to(device)
X_train,y_train,X_test,y_test = X_train.to(device),y_train.to(device),X_test.to(device),y_test.to(device)
```

# Set manual seed
`torch.manual_seed(42)`

# Set train and evaluation modes of the model in the training loop and testing loop respectively
```
model.train()
model.eval()
```

# Reduce the dimension of the ouptut tensor when calculating the loss
```
y_logits=model0(X_train).squeeze()
loss = loss_fn(y_logits, y_train)
```

# Use logits to calculate the classification loss function
## Binary classification
```
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(y_logits, y_train)
```
## Multiclass classification
```
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(y_logits, y_train)
```

# Use DataLoader to turn dataset to iterables of mini-batches
The shape of dataloader is 4 dimensional, the first dimension is batch dimension
```
train_dataloader = DataLoader(train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True 
)
```
## Number of batches
`len(train_dataloader)`
## Total number of samples
`len(train_dataloader.dataset)`
## Number of samples per batch
```
for X, y in test_dataloader:
  print (len(X), len(y))
```
# model.parameters vs. model.parameters()
`model.parameters` returns the attributes of the model, `model.parameters()` is a method of model that returns an iterator over all the parameters of the model.
`model.parameters` is used to inspect the hyperparameters of the model, such as number of hidden layers and number of neurons(hidden units) in each layer
`model.parameters()` is used as a parameter in optimizer: 
`optimizer =torch.optim.Adam(model1.parameters(),lr=0.01)`

```
model1.parameters
<bound method Module.parameters of spiralsmodel(
  (stack): Sequential(
    (0): Linear(in_features=2, out_features=10, bias=True)
    (1): ReLU()
    (2): Linear(in_features=10, out_features=10, bias=True)
    (3): ReLU()
    (4): Linear(in_features=10, out_features=3, bias=True)
  )
)>
model1.parameters()
<generator object Module.parameters at 0x7ce39c2b14d0>
```
# model.parameters vs. model.state_dict
model.parameters is used to inspect the hyperparameters of the model, while model.state_dict is used to inspect the weights and bias

# nn.Flatten layer expects the input imaging data to be 4-dimensional (N, C, H, W)
Two ways to turn 3D imaging data to 4D imaging data:
1. Use DataLoader to turn imaging dataset to batches: DataLoader(dataset, batch_size=32, shuffle=True)
2. Use unsqueeze(dim=0) to add batch dimention: image.unsqueeze(dim=0)
