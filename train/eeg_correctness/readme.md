# Our EEG training on THINGS-EEG

1. Edit `data_download.py` for choosing number of subjects. Launch this script to download CLIP vectors for images and subjects' EEGs from [dongyangli-del](https://github.com/dongyangli-del/EEG_Image_decode) huggling face hub.
2. Use `config_example.yaml` to configure learning process, EEG Encoder params, paths to downloaded data.
3. Launch `train_eeg_only_contrastive.py --config_path <your config path>` . The script uses *lightning* and can be run on any hardware. The script is encoder learning to map EEG to corresponding image's CLIP vector; the CLIP loss on batch is optimized.