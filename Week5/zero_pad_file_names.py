import os
import re


def pad_filenames(directory, digits=6):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("frame") and file.endswith(".jpg"):
                match = re.match(r"frame(\d+).jpg", file)
                if match:
                    num = int(match.group(1))
                    new_name = f"frame{num:0{digits}d}.jpg"
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, new_name)
                    if old_path != new_path:
                        os.rename(old_path, new_path)
                        print(f"Renamed {file} to {new_name}")


# Example usage:
pad_filenames("/ghome/c5mcv08/C6/week5/CVMasterActionRecognitionSpotting/tracking/398x224")
