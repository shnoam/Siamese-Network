import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt


class SiameseDataset(Dataset):
    def __init__(self, x1_train, x2_train, y_train):
        self.x1_train = x1_train
        self.x2_train = x2_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        x1 = self.x1_train[idx]
        x2 = self.x2_train[idx]
        y = self.y_train[idx]
        return x1, x2, y


class SiameseNetwork(nn.Module):
    def __init__(self, dropout_rate):
        super(SiameseNetwork, self).__init__()
        self.set_seed(316139070)
        self.seed = 316139070
        self.model = self.build_network()
        self.distance_layer = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def set_seed(self,my_seed):
        torch.manual_seed(my_seed)
        np.random.seed(my_seed)

    def _init_weights_and_biases(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)

            if module.bias is not None:
                nn.init.normal_(module.bias, mean=0.5, std=0.01)

    def load_data(self, validation_size=0.2, train=True, analyze=False):
        if train:
            with open('data/Train.pickle', 'rb') as f:
                x1_train, x2_train, y_train, names = pickle.load(f)
            x1_train, x2_train, y_train = np.array(x1_train),np.array(x2_train), np.array(y_train)
            x1_train = torch.tensor(x1_train).float().unsqueeze(1).squeeze(-1)
            x2_train = torch.tensor(x2_train).float().unsqueeze(1).squeeze(-1)
            y_train = torch.tensor(y_train).float().unsqueeze(-1)

            x_train_0, x_val_0, x_train_1, x_val_1, y_train, y_val = train_test_split(
                x1_train, x2_train, y_train, test_size=validation_size, random_state=self.seed
            )

            siamese_train_dataset = SiameseDataset(x_train_0, x_train_1, y_train)
            siamese_val_dataset = SiameseDataset(x_val_0, x_val_1, y_val)
            return siamese_train_dataset, siamese_val_dataset
        else:
            with open('data/Test.pickle', 'rb') as f:
                x1_test, x2_test, y_test, names = pickle.load(f)
            x1_test, x2_test, y_test = np.array(x1_test), np.array(x2_test), np.array(y_test)
            if analyze:     # no need to transform into tensors
                return x1_test, x2_test, y_test, names

            x1_test = torch.tensor(x1_test).float().unsqueeze(1).squeeze(-1)
            x2_test = torch.tensor(x2_test).float().unsqueeze(1).squeeze(-1)
            y_test = torch.tensor(y_test).float().unsqueeze(-1)

            siamese_test_dataset = SiameseDataset(x1_test, x2_test, y_test)
            return siamese_test_dataset
    def build_network(self):
        model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(10, 10)), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=(7, 7)), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=(4, 4)), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=(4, 4)), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Flatten(), nn.Linear(9216, 4096), nn.Sigmoid())    # 256*6*6 according to article

        # initialize weights and biased according to article
        model.apply(self._init_weights_and_biases)

        return model

    def forward(self, twin1, twin2):
        output1 = self.model(twin1)
        output2 = self.model(twin2)
        distance = torch.abs(output1-output2)   # calculate distance between twins
        distance = self.dropout(distance)
        prediction = self.distance_layer(distance)
        return prediction

    def fit(self, epochs, batch_size, threshold, learning_rate, wd, min_delta, patience=3):
        """
        :param epochs:
        :param batch_size:
        :param threshold: to decide whether a prediction is same person or not
        :param learning_rate:
        :param wd: weight decay for L2 regularization
        :param min_delta: used for calculation of stopping criteria
        :param patience: maximal number of epochs without change in loss
        :return: train_loss_per_batch, train_loss_per_epoch, val_loss_per_epoch, train_accuracy_per_epoch, val_accuracy_per_epoch, last_epoch
        """
        siamese_train_dataset, siamese_val_dataset = self.load_data(train=True)
        siamese_train_loader = DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True)
        siamese_val_loader = DataLoader(siamese_val_dataset, batch_size=batch_size, shuffle=False)
        # Set the model to training mode
        train_loss_per_batch = {}
        val_loss_per_epoch = {}
        val_accuracy_per_epoch = {}
        train_accuracy_per_epoch = {}
        train_loss_per_epoch = {}

        criterion = torch.nn.BCELoss()
        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=wd)
        batch_total_train = 0
        batch_total_val = 0
        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):

            self.train()
            train_loss = 0
            train_correct = 0
            num_batches = len(siamese_train_loader)
            for batch_idx, (twin1_batch, twin2_batch, y_batch) in enumerate(siamese_train_loader):
                # twin1_batch = twin1_batch   #.to(device)
                # twin2_batch = twin2_batch   #.to(device)
                # y_batch = y_batch   #.to(device)

                pred = self(twin1_batch, twin2_batch)
                train_predictions_class = (val_pred > threshold).float()
                # Convert predictions to class labels (0 or 1)
                # Class 0 if prediction <= threshold, else class 1
                loss = criterion(pred, y_batch)
                train_correct += torch.sum((train_predictions_class.squeeze() == y_batch.squeeze()).float()).item()
                optimizer.zero_grad()
                loss.backward()
                # Update model parameters
                optimizer.step()
                train_loss_per_batch[batch_total_train] = loss.item()
                train_loss += loss.item()
                batch_total_train += 1
            train_loss_avg = train_loss / num_batches
            train_loss_per_epoch[epoch] = train_loss_avg
            curr_train_accuracy = train_correct / len(siamese_train_dataset)
            train_accuracy_per_epoch[epoch] = curr_train_accuracy
            last_epoch = epochs
            # Validate the model (you can also use a separate validation function)
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_correct = 0
                loss_per_epoch = 0
                for batch_idx, (twin1_batch, twin2_batch, y_batch) in enumerate(siamese_val_loader):
                    # twin1_batch = twin1_batch   #.to(device)
                    # twin2_batch = twin2_batch   #.to(device)
                    # y_batch = y_batch   #.to(device)

                    val_pred = self(twin1_batch, twin2_batch)   # forward pass
                    # get prediction using threshold
                    val_predictions_class = (val_pred > threshold).float()  # Convert predictions to class labels (0 or 1)

                    val_loss = criterion(val_pred, y_batch)     #calculate avg loss of samples in batch

                    val_correct += torch.sum((val_predictions_class.squeeze() == y_batch.squeeze()).float()).item() # Update correct count
                    batch_total_val += 1
                    loss_per_epoch += val_loss * y_batch.size(0)        # multiply by batch size to get total loss value

                # Calculate accuracy and loss for monitoring
                val_loss_per_epoch[epoch] = loss_per_epoch / len(siamese_val_dataset)
                curr_val_accuracy = val_correct / len(siamese_val_dataset)
                val_accuracy_per_epoch[epoch] = curr_val_accuracy

                if patience is not None:   # apply stopping criteria
                    if loss_per_epoch < best_val_loss - min_delta:
                        best_val_loss = loss_per_epoch
                        current_patience = 0
                    else:
                        current_patience += 1
                        if current_patience >= patience:    #rechead maximal epochs of non changing
                            print(f"Early stopping after {epoch} epochs...")
                            last_epoch = epoch
                            break
        return train_loss_per_batch, train_loss_per_epoch, val_loss_per_epoch, train_accuracy_per_epoch, val_accuracy_per_epoch, last_epoch

    def test(self, threshold):
        siamese_test_dataset = self.load_data(train=False)
        siamese_test_loader = DataLoader(siamese_test_dataset, shuffle=False)
        self.eval()  # Set the model to evaluation mode
        test_correct = 0  # Counter for correct predictions
        all_preds = []  # used in analysis
        with torch.no_grad():
            for _, (twin1, twin2, y) in enumerate(siamese_test_loader):
                # Forward pass
                # twin1 = twin1.to(device)
                # twin2 = twin2.to(device)
                # y=y.to(device)
                pred = self(twin1, twin2)
                all_preds.append(pred.item())   # used in analysis
                predictions_class = (pred > threshold).float()

                test_correct += (1 if predictions_class == y else 0)

        # Calculate accuracy
        test_accuracy = test_correct / len(siamese_test_dataset)
        return test_accuracy, np.array(all_preds)

    def analyze_results(self, preds):
        """
        plots the image comparison
        :param preds: prediction array from test set
        """
        x1_test, x2_test, y_test, names = self.load_data(train=False, analyze=True)
        results_dict = {}

        for index, name in enumerate(names):
            y_pair = y_test[index]
            predicted_value = preds[index]

            if y_pair == 0 and predicted_value <= 0.5:  # Different people (actual)  TN
                results_dict['TN_rate'] = predicted_value
                results_dict['TN_names'] = name
            elif y_pair == 0 and predicted_value > 0.5:  # FP
                results_dict['FP_rate'] = predicted_value
                results_dict['FP_names'] = name
            elif y_pair == 1 and predicted_value > 0.5:  # TP
                results_dict['TP_rate'] = predicted_value
                results_dict['TP_names'] = name
            elif y_pair == 1 and predicted_value <= 0.5:  # FN
                results_dict['FN_rate'] = predicted_value
                results_dict['FN_names'] = name
        print(
            f'TN: {results_dict["TN_names"]} \nProbability: {results_dict["TN_rate"]}')
        print("*************************")
        print(
            f'FP: {results_dict["FP_names"]} \nProbability: {results_dict["FP_rate"]}')
        print("*************************")
        print(
            f'TP: {results_dict["TP_names"]} \nProbability: {results_dict["TP_rate"]}')
        print("*************************")
        print(
            f'FN: {results_dict["FN_names"]} \nProbability: {results_dict["FN_rate"]}')
        print("*************************")

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))


        axs[0, 0].imshow(x1_test[names.index(results_dict["TN_names"])], cmap="gray")
        axs[0, 1].imshow(x2_test[names.index(results_dict["TN_names"])], cmap="gray")
        axs[0, 0].set_title('True Negative - correct classification of different people', fontweight='bold',
                            fontsize=16)
        axs[0, 0].axis('off')
        axs[0, 1].axis('off')


        axs[1, 0].imshow(x1_test[names.index(results_dict["FP_names"])], cmap="gray")
        axs[1, 1].imshow(x2_test[names.index(results_dict["FP_names"])], cmap="gray")
        axs[1, 0].set_title('False Positive - wrong classification of different people', fontweight='bold', fontsize=16)
        axs[1, 0].axis('off')
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(x1_test[names.index(results_dict["TP_names"])], cmap="gray")
        axs[0, 1].imshow(x2_test[names.index(results_dict["TP_names"])], cmap="gray")
        axs[0, 0].set_title('True Positive - correct classification of same person', fontweight='bold', fontsize=16)
        axs[0, 0].axis('off')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(x1_test[names.index(results_dict["FN_names"])], cmap="gray")
        axs[1, 1].imshow(x2_test[names.index(results_dict["FN_names"])], cmap="gray")
        axs[1, 0].set_title('False Negative - wrong classification of same person', fontweight='bold', fontsize=16)
        axs[1, 0].axis('off')
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.show()