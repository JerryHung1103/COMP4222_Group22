
from featurizer import get_GraphSAGE_embedding_feature
from graph_sage_embedding import GraphSAGE
from graph_sage_embedding import length_of_embedding
from keras.utils import to_categorical
from model import CNN

import sys
sys.path.append('../GNN/') 
from data_preprocessing import MoleculeDataset


train = MoleculeDataset(root="../data/", filename="HIV_train_oversampled.csv")
test = MoleculeDataset(root="../data/", filename="test.csv", test=True)

model = GraphSAGE(in_channels=train.num_node_features, hidden_channels=128, out_channels=length_of_embedding)
print("wait for converting graph to embedding . . .")
X_train, y_train = get_GraphSAGE_embedding_feature(model, train)
X_test, y_test = get_GraphSAGE_embedding_feature(model, test)

print("shape of X_train:", X_train.shape)
print("shape of y_train:", y_train.shape)

print("shape of X_test:", X_test.shape)
print("shape of y_test:", y_test.shape)


X_train=X_train.detach().numpy()
y_train = to_categorical(y_train, 2)

X_test=X_test.detach().numpy()
y_test = to_categorical(y_test, 2)


CNN.fit(X_train, y_train, batch_size=128, validation_data=(X_test, y_test), epochs=100)

loss, accuracy = CNN.evaluate(X_test, y_test)

print(f'Test final loss: {loss}')
print(f'Test final accuracy: {accuracy}')


