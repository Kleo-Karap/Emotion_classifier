# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:32:06 2023

@author: kleop
"""

import os

# Define the class names
class1 = "Calm"
class2 = "Angry"

# Specify the folder path
folder_path = "C:/Users/kleop/Documents/repos/Ergasia/Classification_results"

# Specify the output file path
output_file = "results.txt"


# Open the output file in write mode
with open(output_file, 'w') as file_out:

    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            
            # Read the classification results from the text file
            with open(file_path, 'r') as file:
                classification_results = file.read()

            # Split the classification results by the separator '-------------------------------------------'
            reports_and_predictions = classification_results.split("-------------------------------------------\n")

            # Remove the empty string at the end of the list
            reports_and_predictions = reports_and_predictions[:-1]

            # Initialize counters for class 1 and class 2
            class1_count = 0
            class2_count = 0

            # Count the occurrences of each class in the predictions
            for item in reports_and_predictions:
                if "Prediction: {}".format(class1) in item:
                    class1_count += 1
                elif "Prediction: {}".format(class2) in item:
                    class2_count += 1

            # Determine the majority and minority classes
            if class1_count >= class2_count:
                majority_class = class1
                minority_class = class2
                majority_count = class1_count
                minority_count = class2_count
            else:
                majority_class = class2
                minority_class = class1
                majority_count = class2_count
                minority_count = class1_count

            # Calculate the percentage of the majority and minority classes
            total_count = len(reports_and_predictions)
            majority_percentage = (majority_count / total_count) * 100
            minority_percentage = (minority_count / total_count) * 100

            # Write the results for the current file to the output file
            file_out.write("File: {}\n".format(file_name))
            file_out.write("Percentage of the majority class ({}): {:.2f}%\n".format(majority_class, majority_percentage))
            file_out.write("Percentage of the minority class ({}): {:.2f}%\n".format(minority_class, minority_percentage))
            file_out.write("-------------------------------------------\n")