import os

def load_matched_objects(matched_file):
    """ Load object IDs and camera IDs from matched_objects.txt """
    matched_pairs = set()
    with open(matched_file, 'r') as f:
        for line in f:
            values = line.strip().split(",")
            if len(values) < 6:
                continue  # Skip invalid lines
            
            frame_id = int(values[0])
            cam1_id = int(values[1])
            obj1_id = int(values[2])
            cam2_id = int(values[3])
            obj2_id = int(values[4])

            # Store (frame, camera, object) for both cams
            matched_pairs.add((frame_id, cam1_id, obj1_id))
            matched_pairs.add((frame_id, cam2_id, obj2_id))

    return matched_pairs


def filter_detections(camera_id, sequence):
    """ Filters detections based on matched object IDs and writes to a new file """
    detection_file = f"output/detection/{sequence}/detections_{camera_id}.txt"
    matched_file = f"matched_objects_{sequence}.txt"
    output_dir = f"output/filtered_detection/{sequence}"
    output_file = f"{output_dir}/filtered_detections_{camera_id}.txt"

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    matched_objects = load_matched_objects(matched_file)

    with open(detection_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            values = line.strip().split(",")
            if len(values) < 7:
                continue  # Skip invalid lines

            frame_id = int(values[0])  # Extract frame ID
            object_id = int(values[1])  # Extract object ID
            score = float(values[6])  # Extract score

            # Check if (frame_id, camera_id, object_id) exists in matched_objects AND score > 0.7
            if (frame_id, camera_id, object_id) in matched_objects and score > 0.8:
                f_out.write(line)  # Write only matched detections

    print(f"Filtered detections saved to {output_file}")


# Example Usage
sequence = "S03"  # Change this as needed
for cam in range(10, 16):  # Assuming cameras 1-5
    filter_detections(cam, sequence)
