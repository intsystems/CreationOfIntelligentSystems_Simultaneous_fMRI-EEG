import cv2
import os
import glob

def change_fps_and_save_frames(video_path, output_dir, target_fps, remove_previous_files=True):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove previous files in output directory
    if remove_previous_files:
        files = glob.glob(os.path.join(output_dir, '*'))
        for f in files:
            os.remove(f)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get original FPS and frame count
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original FPS: {original_fps}, Total Frames: {total_frames}, Duration: {total_frames/original_fps}")

    # Calculate the interval to skip frames
    frame_interval = int(original_fps / target_fps)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Only save frames that match the target FPS
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Saved {saved_frame_count} frames to {output_dir}")

# Example usage
# key = 'dme'
# video_path = f'./natview/data/stimuli/{key}.avi'  # Replace with your video path
# output_dir = f'./natview/data/stimuli/{key}'            # Replace with your desired output directory
# target_fps = 3                          # Set your target FPS

# change_fps_and_save_frames(video_path, output_dir, target_fps, remove_previous_files=True)

def rename_files_in_directory(directory_path):
    videoname2key = {
        'Despicable_Me_720x480_English': 'dme',
        'Despicable_Me_720x480_Hungarian': 'dmh',
        'The_Present_720x480': 'tp',
        'Inscapes_02': 'inscapes',
        'Movie1': 'monkey1',
        'Movie2': 'monkey2',
        'Movie3': 'monkey5',
    }
    
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Get the full path to the file
        file_path = os.path.join(directory_path, filename)

        # Check if it is a file
        if os.path.isfile(file_path):
            # Extract the file name without extension
            name, ext = os.path.splitext(filename)

            # Check if the file name is in the dictionary
            if name in videoname2key:
                # Form the new file name
                new_name = videoname2key[name] + ext
                new_file_path = os.path.join(directory_path, new_name)

                # Rename the file
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    directory_path = './natview/data/stimuli'
    rename_files_in_directory(directory_path)

    target_fps = 3
    for key in ['dme', 'dmh', 'tp', 'inscapes', 'monkey1', 'monkey2', 'monkey5']:
        video_path = f'./natview/data/stimuli/{key}.avi'
        output_dir = f'./natview/data/stimuli/{key}'
        change_fps_and_save_frames(video_path, output_dir, target_fps, remove_previous_files=True)