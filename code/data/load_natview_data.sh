#!/bin/bash

# Download fMRI-EEG data for all subs: sub-01, ..., sub-22
python load_natview_subs.py

echo "fMRI-EEG data downloading completed."

# Unpack all .tar.gz files in the directory
for file in ./natview/data/sub-*.tar.gz; do
    if [[ -f "$file" ]]; then  # Check if the file exists
        echo "Unpacking $file..."
        tar -xzvf "$file" -C ./natview/data # Extract the .tar.gz file
    else
        echo "No .tar.gz files found."
    fi
done

echo "fMRI-EEG data unpacking completed."

# Create stimuli directory if does not exist
mkdir -p ./natview/data/stimuli

# Download stimuli data
wget -q https://fcon_1000.projects.nitrc.org/indi/retro/NAT_VIEW/video.tar.gz -P ./natview/data/stimuli
wget -q https://zenodo.org/records/4623809/files/Movie1.avi -P  ./natview/data/stimuli
wget -q https://zenodo.org/records/4623809/files/Movie2.avi -P  ./natview/data/stimuli
wget -q https://zenodo.org/records/4623809/files/Movie3.avi -P  ./natview/data/stimuli

echo "Stimuli downloading completed."

# Unpack .tar.gz stimuli
tar -xvzf ./natview/data/stimuli/video.tar.gz -C ./natview/data/stimuli

echo "Stimuli unpacking completed."

# Find and delete all .tar.gz files in the directory and its subdirectories
find ./natview/data -type f -name "*.tar.gz" -exec rm -f {} +

# Print a message indicating completion
echo "All .tar.gz files in the directory ./natview/data and its subdirectories have been deleted."