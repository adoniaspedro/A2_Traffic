import pandas as pd 
import os
import re 
import warnings 
warnings.filterwarnings('ignore')
 


csv_dir = os.path.join(os.getcwd(), 'Extract')

# List of columns to drop
columns_to_drop = [
    'CD_MELWAY',
    'NB_LATITUDE',
    'NB_LONGITUDE',
    'HF VicRoads Internal',
    'VR Internal Stat',
    'VR Internal Loc',
    'NB_TYPE_SURVEY'
]

# List CSV files in the directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

# Directory to save the modified files
modified_dir = os.path.join(csv_dir, 'modified_csvs')
os.makedirs(modified_dir, exist_ok=True)

# Process each CSV file
for csv_file in csv_files:
    file_path = os.path.join(csv_dir, csv_file)
    print(f"\nProcessing file: {csv_file}")  # Log the file being processed
    
    # Check if the file is empty
    if os.path.getsize(file_path) == 0:
        print(f"Skipping empty file: {csv_file}")
        continue

    try:
        df = pd.read_csv(file_path)

        # Check if the dataframe is empty after loading
        if df.empty:
            print(f"Warning: {csv_file} is empty after reading.")
            continue

        # Strip any leading/trailing spaces from the column names
        df.columns = df.columns.str.strip()

        # Log the actual columns before attempting to drop
        print(f"Columns in file before dropping: {df.columns.tolist()}")

        # Drop the specified columns
        df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

        # Log the columns after dropping
        print(f"Columns in file after dropping: {df_cleaned.columns.tolist()}")

        # Save the cleaned CSV file to the modified directory
        output_file_path = os.path.join(modified_dir, csv_file)
        df_cleaned.to_csv(output_file_path, index=False)
        print(f"Saved cleaned file: {csv_file}")

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

print("Processing complete.")


 
'''
# Load the CSV file
file_path = 'scats_data_w.csv'  # Update this to the actual CSV file path
df = pd.read_csv(file_path)  # Use read_csv instead of read_excel for CSV files

# Assuming 'Location' contains direction info and 'SCATS Number' contains SCAT numbers
direction_column = 'Location'  # Replace with the actual column name containing direction/location data
scats_column = 'SCATS Number'  # Replace with the actual column name containing SCAT numbers

columns_to_drop = [ 
    'CD_MELWAY', 
    'NB_LATITUDE', 
    'NB_LONGITUDE',                
    'HF VicRoads Internal',              
    'VR Internal Stat', 
    'VR Internal Loc', 
    'NB_TYPE_SURVEY'
    ] 

# Define a function to filter by strict direction using regular expressions
def filter_by_strict_direction(df, direction, column):
    # Create a regex pattern that matches the direction at the beginning or end of the string
    # or when it appears as a standalone word (e.g., " N " or " N of " or "Road N")
    direction_pattern = rf'(^|\s){direction}(\s|$|of|Rd|Road|Street)'
    
    # Use regex matching with case insensitivity
    return df[df[column].str.contains(direction_pattern, case=False, na=False, regex=True)]

# List of directions to filter (N = North, S = South, E = East, W = West)
directions = ['N', 'S', 'E', 'W']

# Ensure the output directory exists
output_dir = 'scat_direction_csvs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Group data by SCAT number
for scats_number, group in df.groupby(scats_column):
    # Filter for each direction within the group of SCAT number
    for direction in directions:
        # Filter the group data by the current direction
        direction_data = filter_by_strict_direction(group, direction, direction_column) 
        
        direction_data = direction_data.drop(columns=columns_to_drop, errors='ignore')
        
        # Define the filename using the SCAT number and direction (e.g., 1234_n.csv)
        output_file = f'{scats_number}_{direction.lower()}.csv'
        output_path = os.path.join(output_dir, output_file)
        
        # Save to CSV if data for the direction exists
        if not direction_data.empty:
            direction_data.to_csv(output_path, index=False)
            print(f'Saved data for SCATS Number {scats_number} and direction {direction} to {output_file}')
        else:
            print(f'No data found for SCATS Number {scats_number} and direction {direction}.')

print("CSV files have been saved.")
'''


'''  
# Assuming there's a column that contains direction info, e.g., 'Location'
direction_column = 'Location'  # Replace this with the actual column name containing location data

# Define a function to filter by direction using regular expressions for more precise matching
def filter_by_strict_direction(df, direction, column):
    # Create a pattern that looks for the direction appearing in specific places
    # Example: ' N ' would match ' N ' as a separate word, ' N of ', or ' WARRIGAL_RD N '
    direction_pattern = rf'(^|\s){direction}(\s|$|of|Rd|Street)'
    
    # Use regex matching with case insensitivity
    return df[df[column].str.contains(direction_pattern, case=False, na=False, regex=True)]

# Create files for each direction (North, South, East, West)
directions = ['N', 'S', 'E', 'W']

for direction in directions:
    direction_data = filter_by_strict_direction(df, direction, direction_column)
    
    # Define the filename using the direction
    output_file = f'scats_data_{direction.lower()}.csv'
    
    # Save to CSV if data for the direction exists
    if not direction_data.empty:
        direction_data.to_csv(output_file, index=False)
        print(f'Saved data for {direction} direction to {output_file}')
    else:
        print(f'No data found for {direction} direction.')
'''  
        
'''  
# Group by 'SCATS Number' and save each group to a separate file
for scats_number, group in df.groupby('SCATS Number'):
    # Define the filename using the SCATS Number
    output_file = f'scats_data_{scats_number}.csv'
    # Save to CSV
    group.to_csv(output_file, index=False)

    print(f'Saved data for SCATS Number {scats_number} to {output_file}')
'''

'''
# Loop over unique 'SCATS Number' and 'Location'
for (scats_number, location), group in df.groupby(['SCATS Number', 'Location']):
    # Create the "15 minutes" column (you can adjust this part to generate the time as needed)
    group['15 minutes'] = pd.date_range(start='2006-10-01', periods=len(group), freq='15T')

    # Create a new DataFrame with the required columns
    new_df = pd.DataFrame({
        'scat number': scats_number,
        'location': location,
        '15 minutes': group['15 minutes'],
        'lane 1 flow (veh/15 minutes)': group['V'],  # 'V' is the correct column for lane flow
        '# Lane Points': 1,  # All values set to 1
        '% Observed': '100%'  # All values set to '100%'
    })

    # Define the filename using SCATS Number and Location
    output_file = f"scats_data_{scats_number}_{location.replace(' ', '_')}.csv"

    # Save to CSV
    new_df.to_csv(output_file, index=False)

    print(f'Saved data for SCATS Number {scats_number} and Location {location} to {output_file}')

'''
