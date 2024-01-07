import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt


class FileBrowserApp:
    def __init__(self, root):
        # initialize the app instance
        self.root = root
        self.root.title("DSC Data File Browser")

        # Dictionary to store file paths in groups
        self.groups = {}
        # Variable to store the current group name
        self.current_group = tk.StringVar()
        self.current_group.set("Group_1") # Default group name

        self.group_entry = tk.Entry(root, textvariable=self.current_group, bg='white')
        self.group_entry.pack(pady=100)

        # Label to display selected file paths
        self.file_path_label = tk.Label(root, text="Selected Files:")
        self.file_path_label.pack(pady=10)

        self.browse_button = tk.Button(root, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)

        self.add_data_button = tk.Button(root, text="Add Data for Another Heating Rate", command=self.add_data)
        self.add_data_button.pack(pady=10)

        self.update_file_path_label()

    def browse_file(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_paths:
            group_key = self.current_group.get()
            if group_key not in self.groups:
                self.groups[group_key] = []
            for i, file_path in enumerate(file_paths, start=len(self.groups[group_key]) + 1):
                file_name = f"{group_key}_{i}"
                self.groups[group_key].append((file_name, file_path))
            self.update_file_path_label()

    def add_data(self):
        new_group_key = f"Group_{len(self.groups) + 1}"
        self.current_group.set(new_group_key)
        file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_paths:
            if new_group_key not in self.groups:
                self.groups[new_group_key] = []
            for i, file_path in enumerate(file_paths, start=1):
                file_name = f"{new_group_key}_{i}"
                self.groups[new_group_key].append((file_name, file_path))
            self.update_file_path_label()

    def update_file_path_label(self):
        # Method to update the file path label with the selected files
        text = "Selected Files:\n"
        for group_key, file_paths in self.groups.items():
            if group_key == self.current_group.get():
                text += f"{group_key}:\n"
                for file_name, file_path in file_paths:
                    text += f"  {file_name}: {file_path}\n"
        self.file_path_label.config(text=text)

    def show_files(self):
        self.update_file_path_label()
        self.file_path_label.pack()

    def read_files(self):
        all_contents = {}  # Dictionary to store content from all files
        for group_key, file_paths in self.groups.items():
            print(f"Reading files for {group_key}:")
            group_contents = {}  # Dictionary to store content for the current group
            for file_name, file_path in file_paths:
                try:
                    with open(file_path, 'r', encoding='utf-16') as file:
                        content = file.read()
                        group_contents[f"{file_name}"] = content
                except Exception as e:
                    print(f"Error reading {file_name}: {str(e)}")
            all_contents[group_key] = group_contents

        stored_data = {}  # set a dictionary to store data
        time = {}
        temperature = {}
        heat_flow = {}
        # key-value pair in items (each group key has its value)
        for group_key, group_contents in all_contents.items():  # group_contents are dictionary of the group
            for file_name, content in group_contents.items():
                lines = content.split('\n')
                start_of_data_line = next((i for i, line in enumerate(lines) if line.strip() == "StartOfData"), -1)

                if start_of_data_line != -1:
                    # Extract data columns below "StartofData" line
                    data_lines = '\n'.join(lines[start_of_data_line :])
                    data_array = np.genfromtxt(io.StringIO(data_lines), delimiter='\t', skip_header=True)
                    # Define a key in the desired naming convention
                    key_for_data = f"{file_name}_content"
                    # Store the NumPy array in the dictionary
                    stored_data[key_for_data] = data_array
                    time[key_for_data] = (stored_data[key_for_data])[:,0]
                    temperature[key_for_data] = (stored_data[key_for_data])[:,1]
                    heat_flow[key_for_data] = (stored_data[key_for_data])[:,2]
                    stored_data = {'time': time, 'temperature': temperature, 'heat_flow': heat_flow}
        return stored_data

def baseline(temperature, heat_flow, key_data):
    plt.plot(temperature, heat_flow, color='red', label='Heat Flow over temperature')
    plt.title(f"Please select the baseline start and end points for f{key_data}")
    selected_points = plt.ginput(2, timeout=0, show_clicks=True)
    plt.close()
    selected_x = [point[0] for point in selected_points]
    selected_y = [point[1] for point in selected_points]
    data_point1 = {'temperature': selected_x[0], 'heat_flow': selected_y[0]}
    data_point2 = {'temperature': selected_x[1], 'heat_flow': selected_y[1]}
    plt.plot(temperature, heat_flow, label='Heat Flow vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Heat Flow')
    plt.title(f"DSC Measurement for {key_data}")
    start_index = np.argmin(np.abs(temperature - data_point1['temperature']))
    end_index = np.argmin(np.abs(temperature - data_point2['temperature']))
    plt.plot(temperature, heat_flow)
    plt.scatter(temperature[start_index], heat_flow[start_index], color='green', label='Data Point 1')
    plt.scatter(temperature[end_index], heat_flow[end_index], color='black', label='Data Point 2')
    plt.plot([temperature[start_index], temperature[end_index]], [heat_flow[start_index], heat_flow[end_index]], color='green', linestyle='--', label='Baseline')
    plt.legend()
    plt.show()
    baseline_x = np.linspace(data_point2['temperature'], data_point1['temperature'], num=end_index - start_index)  # Adjust 'num' based on the desired number of points
    baseline_y = np.linspace(data_point2['heat_flow'], data_point1['heat_flow'], num=end_index - start_index)

    return 1

if __name__ == "__main__":
    # initialize the Browser
    root = tk.Tk()
    root.geometry("800x500")
    app = FileBrowserApp(root)
    root.mainloop()
    # input data
    data = app.read_files()
    # pre-process the data
    time = []
    temperature = []
    heat_flow = []
    key_for_data_group = []
    for key_for_data in data.get('time'):
        variable_names = f"time_{key_for_data}"
        globals()[variable_names] = np.array(data.get('time')[key_for_data])
        globals()[variable_names] = (globals()[variable_names])[:700]
        time.append(globals()[variable_names])

    for key_for_data in data.get('temperature'):
        key_for_data_group.append((key_for_data))
        variable_names = f"temperature_{key_for_data}"
        globals()[variable_names] = np.array(data.get('temperature')[key_for_data])
        globals()[variable_names] = (globals()[variable_names])[:700]
        temperature.append(globals()[variable_names])

    for key_for_data in data.get('heat_flow'):
        variable_names = f"heat_flow_{key_for_data}"
        globals()[variable_names] = np.array(data.get('heat_flow')[key_for_data])
        globals()[variable_names] = (globals()[variable_names])[:700]
        heat_flow.append(globals()[variable_names])
    #  Select baseline
    for i in range(0, len(time)):
        baseline(temperature[i],heat_flow[i],key_for_data_group[i])




