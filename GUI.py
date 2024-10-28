import os
import sys
import threading 
import folium
import pandas as pd
import numpy as np
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QTabWidget, QPushButton, QComboBox, QSpinBox, QLabel, QProgressBar, 
                             QFileDialog, QTableWidget, QTableWidgetItem,  QDoubleSpinBox) 
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QUrl
from model import model  # Import the models from model.py
from train import train_model, train_seas  # Import train functions from train.py
from data.data import process_data
from sklearn.metrics import explained_variance_score, r2_score

# Signal class for updating the progress bar from another thread
class WorkerSignals(QObject):
    progress = pyqtSignal(int)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Flow Prediction System")
        self.setGeometry(100, 100, 800, 600)

        # Initialising the UI
        self.setupUi()

        # Add tabs
        self.tabs.addTab(self.create_training_tab(), "Model Training")
        self.tabs.addTab(self.create_adjustment_tab(), "Model Adjustment")
        self.tabs.addTab(self.create_data_tab(), "Data Loading")
        self.tabs.addTab(self.create_evaluation_tab(), "Model Evaluation")
        self.tabs.addTab(self.create_map_tab(), "Map Generation")

        # Set up signals for progress update
        self.signals = WorkerSignals()
        self.signals.progress.connect(self.update_progress)

        # Initialize data file paths
        self.train_file = None
        self.test_file = None
        self.lag = 12  # Set default lag

    def setupUi(self):
        # Set up the tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

    def create_training_tab(self):
        """Create the Model Training tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Add model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "GRU", "SAEs", "Custom"])

        # Add batch size input
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(256)

        # Add epochs input
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(6)

        # Add start training button
        train_button = QPushButton("Start Training")
        train_button.clicked.connect(self.start_training_thread)

        # Add a progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)

        # Add elements to the layout
        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel("Batch Size:"))
        layout.addWidget(self.batch_spin)
        layout.addWidget(QLabel("Epochs:"))
        layout.addWidget(self.epochs_spin)
        layout.addWidget(train_button)
        layout.addWidget(self.progress_bar)

        tab.setLayout(layout)
        return tab

    def create_data_tab(self):
        """Create the Data Loading tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Add buttons to load train and test data
        load_train_button = QPushButton("Load Training Data")
        load_train_button.clicked.connect(self.load_train_file)

        load_test_button = QPushButton("Load Testing Data")
        load_test_button.clicked.connect(self.load_test_file)

        # Add elements to the layout
        layout.addWidget(load_train_button)
        layout.addWidget(load_test_button)
        layout.addWidget(QLabel("Load CSV files for training and testing"))

        tab.setLayout(layout)
        return tab

    def create_evaluation_tab(self):
        """Create the Model Evaluation tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Add the evaluation table
        self.eval_table = QTableWidget()
        self.eval_table.setRowCount(6)
        self.eval_table.setColumnCount(2)
        self.eval_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.eval_table.setVerticalHeaderLabels([
            "Explained Variance", "MAPE (%)", "MAE", "MSE", "RMSE", "R2 Score"
        ])

        # Add the table to the layout
        layout.addWidget(self.eval_table)
        tab.setLayout(layout)
        return tab

    def load_train_file(self):
        """Load training data file using file dialog"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Training Data File", "", 
                                                   "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            self.train_file = file_path
            self.statusBar().showMessage(f"Loaded training file: {file_path}")

    def load_test_file(self):
        """Load testing data file using file dialog"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Testing Data File", "", 
                                                   "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            self.test_file = file_path
            self.statusBar().showMessage(f"Loaded testing file: {file_path}")

    def start_training_thread(self):
        """Start model training in a separate thread"""
        if self.train_file is None or self.test_file is None:
            self.statusBar().showMessage("Please load both training and testing data files before training.")
            return

        self.progress_bar.setValue(0)
        self.train_thread = threading.Thread(target=self.train_model)
        self.train_thread.setDaemon(True)  # Ensure thread does not block app exit
        self.train_thread.start()

    def train_model(self):
        """Actual model training function"""
        model_name = self.model_combo.currentText()
        batch_size = self.batch_spin.value()
        epochs = self.epochs_spin.value()

        # Update the model label
        self.model_label.setText(f"Selected Model: {model_name}")

        # Process the data using the process_data function from data.py
        X_train, y_train, X_test, y_test, _ = process_data(self.train_file, self.test_file, self.lag)

        # Reshape input based on model type
        if model_name in ["LSTM", "GRU"]:
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # 3D for LSTM/GRU
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))      # 3D for LSTM/GRU

        # Select and compile the model
        if model_name == "LSTM":
            m = model.get_lstm([X_train.shape[1], 64, 64, 1])
        elif model_name == "GRU":
            m = model.get_gru([X_train.shape[1], 64, 64, 1])
        elif model_name == "SAEs":
            m = model.get_saes([X_train.shape[1], 400, 400, 400, 1])
        else:
            m = model.get_my_model([X_train.shape[1], 32, 64, 1])

        # Configuration for training
        config = {"batch": batch_size, "epochs": epochs}

        # Train the selected model
        if model_name == "SAEs":
            train_seas(m, X_train, y_train, model_name.lower(), config)
        else:
            train_model(m, X_train, y_train, model_name.lower(), config)

        # Evaluate model and update the evaluation table
        y_pred = m.predict(X_test)
        self.update_evaluation_table(y_test, y_pred)

        # Update progress to 100% after training
        self.signals.progress.emit(100)

    def update_evaluation_table(self, y_true, y_pred):
        """Calculate metrics and update the evaluation table"""
        explained_variance = explained_variance_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = [explained_variance, mape, mae, mse, rmse, r2]

        for i, value in enumerate(metrics):
            self.eval_table.setItem(i, 1, QTableWidgetItem(str(round(value, 4))))

    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)

    def create_adjustment_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()  
        
        # Add Learning Rate adjustment  
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 1.0)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setDecimals(4)

        # Add batch size adjustment
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(256)

        # Add number of layers adjustment
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 10)
        self.num_layers_spin.setValue(3)

        # Add hidden units adjustment
        self.hidden_units_spin = QSpinBox()
        self.hidden_units_spin.setRange(1, 512)
        self.hidden_units_spin.setValue(64)

         # Add a button to apply adjustments
        apply_button = QPushButton("Apply Adjustments")
        apply_button.clicked.connect(self.apply_adjustments)

        # Add elements to the layout
        layout.addWidget(QLabel("Learning Rate:"))
        layout.addWidget(self.learning_rate_spin)
        layout.addWidget(QLabel("Batch Size:"))
        layout.addWidget(self.batch_size_spin)
        layout.addWidget(QLabel("Number of Layers:"))
        layout.addWidget(self.num_layers_spin)
        layout.addWidget(QLabel("Hidden Units per Layer:"))
        layout.addWidget(self.hidden_units_spin)
        layout.addWidget(apply_button)
        
        tab.setLayout(layout)
        return tab  
    
    def apply_adjustments(self): 
        """Apply model adjustments based on user input"""
        learning_rate = self.learning_rate_spin.value()
        batch_size = self.batch_size_spin.value()
        num_layers = self.num_layers_spin.value()
        hidden_units = self.hidden_units_spin.value()

        # Apply these adjustments to the model configuration (example logic)
        self.statusBar().showMessage(
        f"Adjustments applied: Learning Rate = {learning_rate}, Batch Size = {batch_size}, "
        f"Number of Layers = {num_layers}, Hidden Units = {hidden_units}"
        )

        # Update model parameters here or save them for training
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units

 

    def create_map_tab(self):
        """Create the Map Generation tab with route visualization inside the GUI"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Dropdowns for selecting start and end locations
        self.start_location_combo = QComboBox()
        self.end_location_combo = QComboBox()

        # Button to load data for the map
        load_data_button = QPushButton("Load Data for Map")
        load_data_button.clicked.connect(self.load_map_data)

        # Button to show the route on the map
        show_route_button = QPushButton("Show Route")
        show_route_button.clicked.connect(self.display_route)

        # Label to display the selected model
        self.model_label = QLabel("Selected Model: None")

        # WebEngineView to display the HTML map
        self.map_view = QWebEngineView()

        # Add elements to the layout
        layout.addWidget(load_data_button)
        layout.addWidget(QLabel("Select Start Location:"))
        layout.addWidget(self.start_location_combo)
        layout.addWidget(QLabel("Select End Location:"))
        layout.addWidget(self.end_location_combo)
        layout.addWidget(show_route_button)
        layout.addWidget(self.model_label)  # Add the model label to the layout
        layout.addWidget(self.map_view)

        tab.setLayout(layout)
        return tab

    def load_map_data(self):
        """Load the dataset for map visualization and populate dropdowns"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File for Map", "", 
                                               "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                data = pd.read_csv(file_path)

                # Check for required columns in the dataset
                if 'Location' in data.columns and 'NB_LATITUDE' in data.columns and 'NB_LONGITUDE' in data.columns:
                    self.map_data = data  # Store the dataset for mapping
                    locations = data['Location'].unique()

                    # Populate the dropdowns
                    self.start_location_combo.clear()
                    self.end_location_combo.clear()
                    self.start_location_combo.addItems(locations)
                    self.end_location_combo.addItems(locations)

                    self.statusBar().showMessage(f"Loaded data file for map: {file_path}")
                else:
                    self.statusBar().showMessage("The selected file does not have the required columns.")
            except Exception as e:
                self.statusBar().showMessage(f"Error loading file: {e}")


    def display_route(self):
        """Generate and display the route map based on selected start and end locations"""
        if hasattr(self, 'map_data'):
            # Get the selected start and end locations
            start_location = self.start_location_combo.currentText()
            end_location = self.end_location_combo.currentText()

            # Retrieve coordinates for the selected locations
            start_row = self.map_data[self.map_data['Location'] == start_location]
            end_row = self.map_data[self.map_data['Location'] == end_location]

            if not start_row.empty and not end_row.empty:
                start_lat, start_lon = start_row.iloc[0]['NB_LATITUDE'], start_row.iloc[0]['NB_LONGITUDE']
                end_lat, end_lon = end_row.iloc[0]['NB_LATITUDE'], end_row.iloc[0]['NB_LONGITUDE']

                # Create a Folium map centered at the midpoint
                mid_lat, mid_lon = (start_lat + end_lat) / 2, (start_lon + end_lon) / 2
                mymap = folium.Map(location=[mid_lat, mid_lon], zoom_start=13)

                # Add markers for start and end points
                folium.Marker(
                location=[start_lat, start_lon],
                popup=f"<b>Start:</b> {start_location}",
                icon=folium.Icon(color='green', icon='play')
                ).add_to(mymap)

                folium.Marker(
                location=[end_lat, end_lon],
                popup=f"<b>End:</b> {end_location}",
                icon=folium.Icon(color='red', icon='stop')
                ).add_to(mymap)

                # Draw a line between start and end points
                folium.PolyLine(
                locations=[[start_lat, start_lon], [end_lat, end_lon]],
                color='blue', weight=5, opacity=0.7
                ).add_to(mymap)

                # Predict traffic flow
                predicted_flow = self.predict_flow([start_lat, start_lon], [end_lat, end_lon])

                # Display the predicted flow on the map
                if predicted_flow is not None:
                    rounded_flow = round(predicted_flow)
                    popup = f"<b>Predicted Flow:</b> {rounded_flow} vehicles/hour"
                    folium.Marker(
                        location=[end_lat, end_lon],
                        popup=popup,
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(mymap)

                # Save the map as an HTML file
                map_file = 'route_map.html'
                mymap.save(map_file)

                # Convert the file path to a QUrl and display it in QWebEngineView
                qurl = QUrl.fromLocalFile(os.path.realpath(map_file))
                self.map_view.setUrl(qurl)
            else:
                self.statusBar().showMessage("Invalid start or end location selected.")
        else:
            self.statusBar().showMessage("Please load a dataset before displaying the route.")

    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance between two points on the Earth."""
        R = 6371  # Radius of the Earth in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance
    
    def predict_flow(self, start_location, end_location):
        """Predict traffic flow between two locations using a simple heuristic."""
        start_lat, start_lon = start_location
        end_lat, end_lon = end_location
        # Calculating the distance between the two points
        distance = self.haversine(start_lat, start_lon, end_lat, end_lon)
        if distance == 0:
            return 0.0
        # Simple heuristic: flow decreases with distance
        predicted_flow = max(1000 - distance * 10, 0) 
        return predicted_flow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error occurred: {e}")
