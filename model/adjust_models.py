import h5py
import json
import os

# Define the base directory for your model files
base_dir = 'C:/Users/cucum/Downloads/COS30018/TrafficFlowPrediction/model/'

# Define the model file names
model_files = ['lstm.h5', 'gru.h5', 'saes.h5', 'your_mum.h5']

# Loop through each model file and adjust it
for model_file in model_files:
    model_path = os.path.join(base_dir, model_file)

    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        continue

    # Open and modify the model file
    with h5py.File(model_path, 'r+') as f:
        # Read the model configuration
        model_config = f.attrs['model_config']
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')

        # Convert the JSON string to a dictionary
        model_config_dict = json.loads(model_config)

        # Check if 'layers' is a list and iterate over it
        if isinstance(model_config_dict['config'], dict) and 'layers' in model_config_dict['config']:
            layers = model_config_dict['config']['layers']
            if isinstance(layers, list):
                # Iterate through each layer in the list
                for layer in layers:
                    if 'config' in layer and 'batch_shape' in layer['config']:
                        layer['config']['input_shape'] = layer['config'].pop('batch_shape')

        # Convert the dictionary back to a JSON string
        modified_model_config = json.dumps(model_config_dict)

        # Save the modified configuration back into the .h5 file
        f.attrs['model_config'] = modified_model_config

    print(f"Adjusted model file: {model_path}")
