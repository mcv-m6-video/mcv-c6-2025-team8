import numpy as np

def main(det_index):

    # Paths to input and output text files

    txt_path = f"filtered_detections_{det_index}.txt"
    OUT_TXT_PATH = f"filtered_detection_{det_index}.txt"

    # Read the input text file
    with open(txt_path, 'r') as infile:
        rows = infile.readlines()

    seen_tracks = {}

    for row in rows:
        # Assuming space or tab separation between values
        columns = row.strip().split(',') 
        # print(columns) # Modify the delimiter if needed
        frame_id = int(columns[0])
        track_id = int(columns[1])

        x_min, y_min = float(columns[2]), float(columns[3])
        width, height = float(columns[4]), float(columns[5])

        centroid = (x_min + width / 2, y_min + height / 2)

        # Update the track data for each track_id
        if track_id not in seen_tracks:
            seen_tracks[track_id] = {
                "first_frame": frame_id,
                "first_centroid": centroid,
                "last_frame": None,
                "last_centroid": None
            }
        else:
            inner_dict = seen_tracks[track_id].copy()
            inner_dict["last_frame"] = frame_id
            inner_dict["last_centroid"] = centroid
            seen_tracks[track_id] = inner_dict

    # print(seen_tracks)

    tracks_to_maintain = []
    for track_id, track_info in seen_tracks.items():
        centroid_f = track_info["first_centroid"]
        centroid_l = track_info["last_centroid"]

        if not centroid_l:
            continue

        if abs(centroid_l[0] - centroid_f[0]) + abs(centroid_l[1] - centroid_f[1]) > 50:
            tracks_to_maintain.append(track_id)

    print(tracks_to_maintain)

    # Write the filtered tracks to a new text file
    with open(OUT_TXT_PATH, 'w') as outfile:
        for row in rows:
            columns = row.strip().split(',')
            # print(columns)  # Split each line by space or tab
            track_id = int(columns[1])
            # print(track_id)

            if track_id in tracks_to_maintain:
                # print('aads')
                outfile.write(row)


for i in range(1, 4):
    main(i)

