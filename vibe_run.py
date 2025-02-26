import os
import subprocess
import time

# File paths
vibe_exe = r"C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C6_project/ViBe/ViBe/vibe-rgb.exe"
input_file = r"C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C6_project/mcv-c6-2025-team8/AICity_data/AICity_data/train/S03/c010/vdo.avi"

# Output directory
output_dir = r"C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C6_project/mcv-c6-2025-team8/results_avi_2"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

# Log file path
log_file = os.path.join(output_dir, "processing_log.txt")

# Parameters to test
num_samples_list = [10, 20, 30]  # Example values
matching_threshold_list = [10, 15, 20, 25]  # Example values
subsampling_factor_list = [8, 16, 24]  # Example values

# Flag for displaying results only
disp_results_only = True  # Default is False

# Open log file for writing
with open(log_file, "w") as log:
    log.write("ViBe Processing Log\n")
    log.write("=" * 50 + "\n\n")

    # Loop through parameter combinations
    for num_samples in num_samples_list:
        for matching_threshold in matching_threshold_list:
            for subsampling_factor in subsampling_factor_list:
                # Generate output filename based on parameters
                output_filename = f"test2_n{num_samples}_t{matching_threshold}_s{subsampling_factor}.avi"
                output_path = os.path.join(output_dir, output_filename)

                # Construct command
                cmd = [
                    vibe_exe,
                    "-i", input_file,
                    "-n", str(num_samples),
                    "-t", str(matching_threshold),
                    "-s", "2",  # Default: 2 (kept constant)
                    "--subsampling-factor", str(subsampling_factor),
                    "-o", output_path
                ]

                # Add display-only flag if enabled
                if disp_results_only:
                    cmd.insert(1, "-ro")

                # Start timing
                start_time = time.time()

                print(f"Running: {cmd}")
                log.write(f"Running: {cmd}\n")
                subprocess.run(cmd)

                # End timing
                end_time = time.time()
                duration = end_time - start_time

                # Log processing time
                log_entry = f"Completed: {output_filename} | Time: {duration:.2f} seconds\n"
                print(log_entry)
                log.write(log_entry + "\n")

    log.write("\nAll processing complete!\n")

print("\nAll processing complete! Check the output directory and log file for results.")

