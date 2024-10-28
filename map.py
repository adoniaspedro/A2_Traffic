import pandas as pd
import folium
import webbrowser
import os

# Update this to the relative or absolute path to your CSV file
csv_file = 'Extract\consolidated_filtered_data.csv'

# Step 1: Load the CSV file into a DataFrame
try:
    data = pd.read_csv(csv_file)

    # Step 2: Clean up column names by stripping any extra spaces
    data.columns = data.columns.str.strip()

    # Step 3: Verify the column names and ensure they include latitude and longitude
    print("Column names:", data.columns)

    # Check if 'NB_LATITUDE' and 'NB_LONGITUDE' are present
    if 'NB_LATITUDE' in data.columns and 'NB_LONGITUDE' in data.columns:
        # Set the map center and zoom level based on the desired view
        initial_latitude = -37.81  # Example latitude from the image (adjust as needed)
        initial_longitude = 145.03  # Example longitude from the image (adjust as needed)
        initial_zoom = 13  # Adjusted zoom level

        # Initialize the map at the specified starting point and zoom level
        mymap = folium.Map(location=[initial_latitude, initial_longitude], zoom_start=initial_zoom)

        # Add individual blue markers for each row of the dataset
        for idx, row in data.iterrows():
            location_name = row['Location'] if 'Location' in data.columns else 'Unknown Location'
            latitude, longitude = row['NB_LATITUDE'], row['NB_LONGITUDE']
            popup_text = f"<b>Location:</b> {location_name}<br><b>Latitude:</b> {latitude}<br><b>Longitude:</b> {longitude}"

            # Add a marker with a blue icon
            folium.Marker(
                location=[latitude, longitude],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(mymap)

        # Save the map to an HTML file
        map_file = 'scats_map.html'
        mymap.save(map_file)
        print(f"Map has been saved as '{map_file}'")

        # Open the generated HTML file in the default web browser
        webbrowser.open(f'file://{os.path.realpath(map_file)}')

    else:
        print("The required columns 'NB_LATITUDE' and 'NB_LONGITUDE' were not found in the data.")
except FileNotFoundError:
    print(f"File not found: {csv_file}. Please check the file path and try again.")



