import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# Define the input and output directories
input_folder = 'TrafficFlowPrediction\data\Extract'
train_output_folder = 'TrafficFlowPrediction\data\Extract\Train'
test_output_folder = 'TrafficFlowPrediction\data\Extract\Test'

# Create output directories if they don't exist
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

# Get all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Process each CSV file
for csv_file in csv_files:
    file_path = os.path.join(input_folder, csv_file)

    # Load the data
    data = pd.read_csv(file_path)

    # Create a copy of the data to keep original 'Location' and 'Date' columns for saving
    data_original = data.copy()

    # Encode non-numeric columns ('SCATS Number' and 'Location') for model training
    label_encoder = LabelEncoder()
    data['SCATS Number'] = label_encoder.fit_transform(data['SCATS Number'])
    data['Location'] = label_encoder.fit_transform(data['Location'])

    # Convert 'Lane Points' to a suitable format for training
    data['Lane Points'] = data['Lane Points'].apply(lambda x: eval(x) if isinstance(x, str) else x)  # Convert string to list
    data['Lane Points'] = data['Lane Points'].apply(
        lambda x: sum(x) / len(x) if isinstance(x, list) and len(x) > 0 else x if isinstance(x, (int, float)) else 0
    )

    # Define the split index
    split_index = int(len(data) * 0.8)  # Calculate 80% index

    # Split the data into training and testing sets based on the specified rows
    train_data = data.iloc[:split_index]  # First 80% rows
    test_data = data.iloc[split_index:]    # Remaining 20% rows

    # Define features and target for training
    X_train = train_data[['SCATS Number', 'Location']]
    y_train = train_data['Lane Points']

    # Define features and target for testing
    X_test = test_data[['SCATS Number', 'Location']]
    y_test = test_data['Lane Points']

    # Train the RandomForestClassifier model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{csv_file} - Model Accuracy: {accuracy * 100:.2f}%")

    # Save the train and test sets into separate CSV files
    train_original = data_original.iloc[:split_index]
    test_original = data_original.iloc[split_index:]

    # Generate output file names
    train_file_name = f"{os.path.splitext(csv_file)[0]}_train.csv"
    test_file_name = f"{os.path.splitext(csv_file)[0]}_test.csv"

    # Save the train and test files
    train_original.to_csv(os.path.join(train_output_folder, train_file_name), index=False)
    test_original.to_csv(os.path.join(test_output_folder, test_file_name), index=False)

    print(f"Saved {train_file_name} and {test_file_name}")
