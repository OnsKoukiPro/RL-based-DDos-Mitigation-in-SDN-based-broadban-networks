import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class PerformanceAnalyzer:
    def __init__(self, result_file, flowcount_file):
        self.result_file = result_file
        self.flowcount_file = flowcount_file
        self.tp = 0  # True Positives
        self.tn = 0  # True Negatives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.flow_data = []  # Stores flow count over time
        self.attack_data = []  # Stores attack/normal labels over time

    def load_metrics(self):
        # Load the results from the result.csv file
        with open(self.result_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > 0:
                    self.attack_data.append(int(row[-1]))  # Last column indicates attack (1) or normal (0)

        # Load flow count data from flowcount.csv
        with open(self.flowcount_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                if len(row) > 0 and row[1].isdigit():  # Ensure flowcount is numeric
                    self.flow_data.append(int(row[1]))  # Second column is the flow count

    def calculate_performance(self):
        # Calculate True Positives, True Negatives, False Positives, and False Negatives
        for i in range(min(len(self.attack_data), len(self.flow_data))):
            attack = self.attack_data[i]
            flow = self.flow_data[i]
            
            if attack == 1:  # attack detected
                if flow > 100:  # Assuming high flow count represents detected attack
                    self.tp += 1  # True Positive
                    print(f"True Positive at index {i}: Flow {flow}, Attack {attack}")
                else:
                    self.fn += 1  # False Negative
                    print(f"False Negative at index {i}: Flow {flow}, Attack {attack}")
            else:  # normal traffic
                if flow <= 100:  # Assuming low flow count represents normal traffic
                    self.tn += 1  # True Negative
                    #print(f"True Negative at index {i}: Flow {flow}, Attack {attack}")
                else:
                    self.fp += 1  # False Positive
                    #print(f"False Positive at index {i}: Flow {flow}, Attack {attack}")


    def calculate_metrics(self):
        # Calculate precision, recall, and F1-score
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("True Positives:", self.tp)
        print("True Negatives:", self.tn)
        print("False Positives:", self.fp)
        print("False Negatives:", self.fn)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1_score)

    def plot_performance(self):
        # Plot flow count over time
        plt.figure(figsize=(12, 6))
        
        # Plot flow count
        plt.subplot(2, 1, 1)
        plt.plot(self.flow_data, label='Flow Count')
        plt.xlabel('Time')
        plt.ylabel('Flow Count')
        plt.title('Flow Count over Time')
        plt.legend()

        # Plot attack vs normal traffic over time
        plt.subplot(2, 1, 2)
        plt.plot(self.attack_data, label='Attack/Normal Traffic', color='orange')
        plt.xlabel('Time')
        plt.ylabel('Attack/Normal')
        plt.title('Attack vs Normal Traffic over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Path to the result and flowcount CSV files
    result_file = 'result.csv'
    flowcount_file = 'switch_1_flowcount.csv'  # Replace with the correct path to your flow count file

    # Initialize the PerformanceAnalyzer
    analyzer = PerformanceAnalyzer(result_file, flowcount_file)
    
    # Load metrics from the CSV files
    analyzer.load_metrics()

    # Calculate performance metrics
    analyzer.calculate_performance()

    analyzer.calculate_metrics()

    # Plot the performance over time
    analyzer.plot_performance()
