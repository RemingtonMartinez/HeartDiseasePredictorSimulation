import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np
from scipy import stats
import matplotlib.ticker as mticker

# Define your custom dataset class
class HeartDiseaseDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Define your neural network model
class HeartDiseasePredictor(nn.Module):
    def __init__(self, input_dim):
        super(HeartDiseasePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def train(train_loader, model, criterion, optimizer, num_epochs):
    train_losses = []
    validation_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())

            # Reshape the labels to match the shape of outputs
            labels = labels.view(-1, 1).float()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Calculate training loss for this epoch
        with torch.no_grad():
            model.eval()
            train_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())
                labels = labels.view(-1, 1).float()
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_losses.append(train_loss / len(train_loader))
            
            # Calculate training accuracy
            train_accuracy = correct / total
            
        # Calculate validation accuracy for this epoch
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())
                labels = labels.view(-1, 1).float()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            validation_accuracy.append(correct / total)

    return model, train_losses, validation_accuracy

def test(test_loader, model_file):
    model = HeartDiseasePredictor(input_dim).to(device)  # Move model to GPU
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Evaluate the model on the test data
    with torch.no_grad():
        correct = 0
        total = 0
        y_true = []
        y_scores = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())

            # Reshape the labels to match the shape of outputs
            labels = labels.view(-1, 1).float()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

    # Calculate test accuracy
    accuracy = correct / total
    #print(f"Test Accuracy: {accuracy}")

    return y_true, y_scores, accuracy

def plot_loss_curve(train_losses, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png', dpi=2000)

def plot_accuracy_curve(validation_accuracy, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.savefig('accuracy_curve.png', dpi=2000)

def plot_confusion_matrix(y_true, y_scores):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, (np.array(y_scores) > 0.5).astype(int))

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=2000)

def plot_roc_curve(y_true, y_scores, roc_auc):
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png', dpi=2000)

def plot_scatter_plot(y_true, y_scores):
    # Create a scatter plot for predictions vs. true values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_scores, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot: True vs. Predicted Values')

def plot_violin_plot(y_true, y_scores):
    # Create a violin plot with a bar plot inside it
    data_df = pd.DataFrame({'True Values': y_true, 'Predicted Values': y_scores})
    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(data=data_df, x='True Values', y='Predicted Values')
    sns.stripplot(data=data_df, x='True Values', y='Predicted Values', jitter=True, alpha=0.5, color='black', size=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')

    # Calculate the statistical significance between diseased and non-diseased states
    diseased_indices = np.where(np.array(y_true) == 1)[0]
    non_diseased_indices = np.where(np.array(y_true) == 0)[0]

    diseased_scores = np.array(y_scores)[diseased_indices]
    non_diseased_scores = np.array(y_scores)[non_diseased_indices]

    # Perform a statistical significance test (e.g., t-test)
    t_stat, p_value = stats.ttest_ind(diseased_scores, non_diseased_scores)

    # Display the p-value in scientific notation
    p_value_sci = "{:0.2e}".format(p_value)

    # Calculate the midpoint of the violin plots
    midpoint = (np.mean(diseased_scores) + np.mean(non_diseased_scores)) / 2

    # Draw a solid horizontal line connecting the midpoints
    plt.hlines(midpoint, 0.15, .85, colors='black', linestyle='-', lw=1.5)

    # Display the p-value above the line
    plt.text(0.5, midpoint + 0.05, f'T-test; p = {p_value_sci}', ha='center', fontsize=12)

if __name__ == '__main__':
    method = 'GaussianNoiseGenerator'  # Set to 'train' or 'test' depending on what you want to do
    
    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if method == 'train':
        # Load the data from 'heart.csv' using pandas
        data = pd.read_csv('heart.csv')

        # Separate the features (X) and the target variable (y)
        X = data.drop('target', axis=1).values
        y = data['target'].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the input features (you can use other preprocessing techniques)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Data loading and preprocessing for training
        train_dataset = HeartDiseaseDataset(X_train, y_train)
        test_dataset = HeartDiseaseDataset(X_test, y_test)
        
        # Move datasets to GPU if available
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        input_dim = X_train.shape[1]  # Define input_dim based on the number of columns in X_train
        model = HeartDiseasePredictor(input_dim).to(device)  # Move model to GPU
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 100  # Define the number of training epochs

        # Train the model
        model, train_losses, validation_accuracy = train(train_loader, model, criterion, optimizer, num_epochs)

        # Plot the loss curve
        plot_loss_curve(train_losses, num_epochs)

        # Plot the accuracy curve
        plot_accuracy_curve(validation_accuracy, num_epochs)

        # Save the trained model to a .pt file
        torch.save(model.state_dict(), 'heart_disease_model.pt')
        
        # Evaluate the model on the test data
        y_true, y_scores, _ = test(test_loader, 'heart_disease_model.pt')
        
        # Convert numpy arrays to lists
        y_true = [item for sublist in y_true for item in sublist]
        y_scores = [item for sublist in y_scores for item in sublist]
        
        # Compute confusion matrix and plot
        plot_confusion_matrix(y_true, y_scores)

        # Calculate ROC curve and AUC, and plot ROC curve
        roc_auc = roc_auc_score(y_true, y_scores)
        plot_roc_curve(y_true, y_scores, roc_auc)

        # Create a scatter plot for predictions vs. true values
        plot_scatter_plot(y_true, y_scores)
        
        # Create a violin plot with statistical significance
        plot_violin_plot(y_true, y_scores)
        
        plt.show()

    elif method == 'test':
        # Test the model using a different CSV file and pre-trained model
        test_data_file = 'test_data.csv'
        test_data = pd.read_csv(test_data_file)
        
        # Separate the features (X) and the target variable (y)
        X_test = test_data.drop('target', axis=1).values
        y_test = test_data['target'].values
        
        # Standardize the input features (you can use other preprocessing techniques)
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)
        
        # Data loading and preprocessing for testing
        test_dataset = HeartDiseaseDataset(X_test, y_test)
        
        # Move datasets to GPU if available
        test_loader = DataLoader(test_dataset, batch_size=64)
        input_dim = X_test.shape[1]  # Define input_dim based on the number of columns in X_test
        
        # Evaluate the model on the test data
        y_true, y_scores, _ = test(test_loader, 'heart_disease_model.pt')
        
        # Convert numpy arrays to lists
        y_true = [item for sublist in y_true for item in sublist]
        y_scores = [item for sublist in y_scores for item in sublist]
        
        # Compute confusion matrix and plot
        plot_confusion_matrix(y_true, y_scores)
        
    elif method == 'GaussianNoiseGenerator':
        # Step 1: Load the CSV file into a DataFrame
        df = pd.read_csv('heart.csv')

        # Step 2: Identify the 'target' column and store it separately
        target_column = df['target']

        # Initialize an empty list to store the accuracy values for each noise percentage
        accuracies = []

        # Define the percentage range for noise (0% to 100%) in increments of 5%
        noise_percentages = range(0, 101, 5)

        for percentage in noise_percentages:
            # Calculate the noise scale factor for the current percentage
            noise_scale_factor = percentage / 100
    
            # Add Gaussian noise to the DataFrame (excluding the 'target' column)
            noisy_df = df.drop(columns=['target']) + noise_scale_factor * np.random.randn(len(df), len(df.columns) - 1)
    
            # Combine the 'target' column back into the DataFrame
            noisy_df['target'] = target_column
            
            # Separate the features (X) and the target variable (y)
            X_test = noisy_df.drop('target', axis=1).values
            y_test = noisy_df['target'].values
        
            # Standardize the input features (you can use other preprocessing techniques)
            scaler = StandardScaler()
            X_test = scaler.fit_transform(X_test)
        
            # Data loading and preprocessing for testing
            test_dataset = HeartDiseaseDataset(X_test, y_test)
        
            # Move datasets to GPU if available
            test_loader = DataLoader(test_dataset, batch_size=64)
            input_dim = X_test.shape[1]  # Define input_dim based on the number of columns in X_test
        
            # Evaluate the model on the test data
            _,_, Accuracy = test(test_loader, 'heart_disease_model.pt')
            
            # Append the accuracy to the list
            accuracies.append(Accuracy)
            
        # Plot the scatter plot
        plt.figure(figsize=(10, 5))
        plt.scatter(noise_percentages, accuracies, label='Accuracy', marker='o', color='blue')
        plt.xlabel('Noise Percentage')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Noise Percentage')

        # Calculate Pearson correlation coefficient
        corr_coefficient, _ = stats.pearsonr(noise_percentages, accuracies)

        # Plot the line related to Pearson correlation
        x_line = np.linspace(min(noise_percentages), max(noise_percentages), 100)
        y_line = corr_coefficient * (x_line - np.mean(noise_percentages)) / np.std(noise_percentages) + np.mean(accuracies)
        plt.plot(x_line, y_line, label=f'Pearson Correlation Line (r = {corr_coefficient:.2f})', linestyle='--', color='red')

        # Display the Pearson correlation coefficient
        plt.text(0.5, 0.1, f'Pearson Correlation Coefficient = {corr_coefficient:.2f}', ha='center', fontsize=12, transform=plt.gca().transAxes)

        # Add a legend
        plt.legend()

        # Save the plot as an image
        plt.savefig('accuracy_vs_noise_percentage.png', dpi=2000)

        # Show the plot
        plt.show()
        
        # Save the accuracies and noise percentages to a CSV file
        df = pd.DataFrame({'Noise Percentage': noise_percentages, 'Accuracy': accuracies})
        df.to_csv('accuracy_vs_noise_percentage.csv', index=False)

