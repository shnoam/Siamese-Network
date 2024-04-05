import csv
import os
import time
import torch
import data_load
from data_load import *
from network import SiameseNetwork
import matplotlib.pyplot as plt
import pandas as pd
pickle_path = os.path.join('data', 'Train.pickle')
# Check if the file exists

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_the_data_to_pickle():
    if os.path.exists(pickle_path):
        pickle_exist = True  # File exists
    else:
        pickle_exist = False  # File does not exist

    if not pickle_exist: # if we didnt loaded the data yet
        data_load.load_images(set_name='Train',data_path='data',output_path='data/Train.pickle')
        data_load.load_images(set_name='Test',data_path='data',output_path='data/Test.pickle')
    else:
        print('we alraedy loaded the data to pickle')


def plot_metrics(metric_dict,plot_name,title, y_label, x_label, color,folder_name):

    plt.figure(figsize=(10, 5))
    for label, values in metric_dict.items():
        plt.plot(list(values.keys()), list(values.values()), label=label, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    folder_name = "plots/"+folder_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    fig_path = os.path.join(folder_name, f"{plot_name}.png")
    plt.savefig(fig_path)
    plt.show()

def plot_loss_per_epoch(train_loss_values, val_loss_values,plot_name, title,folder_name):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label='Train Loss', color='red', linestyle='-')
    plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label='Validation Loss', color='green', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    fig_path = os.path.join(folder_name, f"{plot_name}.png")
    plt.savefig(fig_path)
    plt.show()


def plot_accuracies_per_epoch(train_acc_values, val_acc_values, plot_name, title, folder_name):
    val_acc_values = [value for value in val_acc_values]
    train_acc_values = [value for value in train_acc_values]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_acc_values) + 1), train_acc_values, label='Train Accuracy', color='red', linestyle='-')
    plt.plot(range(1, len(val_acc_values) + 1), val_acc_values, label='Validation Accuracy', color='green', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    fig_path = os.path.join(folder_name, f"{plot_name}.png")
    plt.savefig(fig_path)
    plt.show()
def plot_last_val_loss(df, folder):

    # Read data from CSV file

    # Plotting the bar plot for 'last_val_loss'
    plt.figure(figsize=(6, 6))
    bars = plt.bar(df.index+1, df['last_val_loss'], color='lightblue', width=0.5)

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), va='bottom')

    plt.xlabel('Model')
    plt.ylabel('Validation Loss')
    plt.title('Comparison of Validation Loss')
    plt.xticks(df.index+1)
    fig_path = os.path.join(folder,"val_loss.png")
    plt.savefig(fig_path)
    plt.show()

def plot_training_time(df, folder):
    # Plotting the bar plot for 'last_val_loss'
    plt.figure(figsize=(6, 6))
    bars = plt.bar(df.index+1, df['training_time'], color='purple', width=0.5)

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), va='bottom')

    plt.xlabel('Model')
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of Training Time')
    plt.xticks(df.index+1)
    fig_path = os.path.join(folder,"training_time.png")
    plt.savefig(fig_path)
    plt.show()

def plot_accuracy_comparison(df, folder):
    # Plotting the bar plot for 'val_accuracy' and 'test_accuracy'
    plt.figure(figsize=(6,6))

    index = df.index
    bar_width = 0.25

    plt.bar((index + 1) - bar_width/2, df['val_accuracy'], bar_width, label='Validation Accuracy', color='green')
    plt.bar((index + 1) + bar_width/2, df['test_accuracy'], bar_width, label='Test Accuracy', color='red')

    # Add value labels on top of each bar
    for i, val_acc, test_acc in zip(index, df['val_accuracy'], df['test_accuracy']):
        plt.text((i + 1) - bar_width/2, val_acc, f'{val_acc:.2f}', ha='center', va='bottom')
        plt.text((i + 1) + bar_width/2, test_acc, f'{test_acc:.2f}', ha='center', va='bottom')

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Validation and Test Accuracy')
    plt.xticks(index + 1)
    plt.legend(loc='lower left')
    plt.tight_layout()

    fig_path = os.path.join(folder, "accuracy.png")
    plt.savefig(fig_path)
    plt.show()

def run_one_experiment( n_epochs,batch_size,threshold,learning_rate,weight_decay,dropout,min_delta,to_analyze=False):

    parameters_name = f'ep_{n_epochs}_bs_{batch_size}_th_{threshold}_lr_{learning_rate}_wd_{weight_decay}_dropout_{dropout}_delta_{min_delta}'
    print(f'Running combination with {parameters_name}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Create Siamese network
    siamese = SiameseNetwork(dropout_rate=dropout)

    siamese.to(device)

    # Fit the model
    start_training_time = time.time()
    train_loss_batch, train_loss_epoch, val_loss_epoch,train_accuracies,val_accuracies, last_epoch= siamese.fit(
        epochs=n_epochs, batch_size=batch_size, threshold=threshold, learning_rate=learning_rate,wd=weight_decay,min_delta=min_delta)
    end_training_time = time.time()
    training_time_seconds = end_training_time - start_training_time
    training_time_minutes = training_time_seconds / 60

    #  test model
    test_acc,preds = siamese.test(threshold=threshold)

    # create plots of model
    folder_name = f'ep_{n_epochs}_bs_{batch_size}_th_{threshold}_lr_{learning_rate}_wd_{weight_decay}_dropout_{dropout}_delta_{min_delta}'

    plot_metrics({'Training Loss': train_loss_batch},"train_loss_per_batch", 'Training Loss per Batch', 'Loss',
                 "Batch", "blue",folder_name)

    plot_loss_per_epoch(train_loss_epoch.values(), val_loss_epoch.values(),"Training_Validation_Loss",
                        'Training VS Validation Loss',folder_name)

    # plot_metrics({"Val Accuracy": val_accuracies}, "Validation_accuracy", "Validation Accuracy", 'Accuracy',
    #              'Epoch', "orange", folder_name)
    plot_accuracies_per_epoch(train_accuracies.values(), val_accuracies.values(),"Training_Validation_Accuracy",
                        'Training VS Validation Accuracy',folder_name)
    if to_analyze:
        siamese.analyze_results(preds)

    return train_loss_epoch, val_loss_epoch, train_accuracies,val_accuracies, test_acc, training_time_seconds, last_epoch

def run_expirements():
    thresholds = [0.5,0.6]
    learning_rates = [0.0001,0.00005]
    n_epochs = [15]
    batch_sizes = [32]
    weight_decays = [0,2e-04,7e-04]
    dropout_values = [0,0.2,0.3]
    min_deltas = [0.05,0.1]
    results = []
    for epochs in n_epochs:
        for batch_size in batch_sizes:
            for th in thresholds:
                for lr in learning_rates:
                    for wd in weight_decays:
                      for drop_val in dropout_values:
                          for delta in min_deltas:
                            train_loss_epoch, val_loss_epoch, train_accuracies, val_accuracies, test_acc,training_time, last_epoch= run_one_experiment(
                                n_epochs=epochs,
                                batch_size=batch_size,
                                threshold =th,
                                learning_rate=lr,
                                weight_decay=wd,
                                dropout=drop_val,
                                min_delta=delta
                            )
                            last_train_loss = list(train_loss_epoch.values())[-1]
                            last_val_loss = list(val_loss_epoch.values())[-1]
                            last_val_accuracy = list(val_accuracies.values())[-1]
                            results.append({
                                'n_epochs': epochs,
                                'batch_size': batch_size,
                                'threshold': th,
                                'learning_rate': lr,
                                'weight_decay': wd,
                                "dropout": drop_val,
                                "min_delta": delta,
                                "last_epoch": last_epoch,
                                'last_train_loss': last_train_loss,
                                'last_val_loss': last_val_loss,
                                'val_accuracy': last_val_accuracy,
                                'test_accuracy': test_acc,
                                "training_time": training_time
                            })


    with open('experiment_results.csv', 'w', newline='') as csvfile:
        fieldnames = ["n_epochs","batch_size","threshold","learning_rate",'weight_decay', 'dropout','min_delta',"last_epoch",'last_train_loss','last_val_loss','val_accuracy', 'test_accuracy',"training_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    return results
def run_images_compare():
    model1 = run_one_experiment(
                                n_epochs=15,
                                batch_size=16,
                                threshold =0.5,
                                learning_rate=5e-05,
                                weight_decay=0,
                                dropout=0.2,
                                min_delta = 0.05,
                                to_analyze=True
                            )
    print("analyze data successfully")
    return model1

def create_comparison_plots(file_name, folder_name):
    result_df = pd.read_csv(file_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plot_training_time(result_df, folder_name)
    plot_last_val_loss(result_df, folder_name)
    plot_accuracy_comparison(result_df, folder_name)

