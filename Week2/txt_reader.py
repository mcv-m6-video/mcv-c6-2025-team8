import re
import numpy as np

def parse_results(file_path):
    # Initialize an empty list to store the results
    results = []

    with open(file_path, 'r') as file:
        content = file.read()

        # Split the content into blocks based on 'Results for' separator
        detection_files = content.split('Results for tracked')[1:]

        for detection in detection_files:
            # Extract the file name (before the first newline)
            file_name = detection.split(':')[0].strip()

            # Extract the HOTA score array
            hota_score_match = re.search(r'HOTA Score:\s*\[([^\]]+)\]', detection)
            if hota_score_match:
                hota_scores = np.array([float(x) for x in hota_score_match.group(1).split()])
                
                # Compute the average HOTA score
                average_hota = np.mean(hota_scores)
            else:
                average_hota = None

            # Extract the Identity Score (IDF1)
            idf1_match = re.search(r'Identity Score \(IDF1\):\s*([0-9.]+)', detection)
            if idf1_match:
                idf1_score = float(idf1_match.group(1))
            else:
                idf1_score = None

            # Store the result as a tuple
            results.append((file_name, average_hota, idf1_score))

    # Sort the results by average HOTA score (highest to lowest)
    results.sort(key=lambda x: x[1], reverse=True)

    return results

def print_results(results):
    # Print the results with numbering
    for idx, (file_name, avg_hota, idf1_score) in enumerate(results, start=1):
        print(f"{idx}. File: {file_name}")
        print(f"   Average HOTA Score: {avg_hota:.4f}")
        print(f"   IDF1 Score: {idf1_score:.4f}")
        print("="*50)

# Usage example
file_path = 'Week2/tracking_results/combined_results_.txt'  # Replace with the actual path to your txt file
results = parse_results(file_path)
print_results(results)

