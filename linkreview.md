# LinkReview

- Here we have collect info about all the works that may be useful for writing our paper
- We divide these works by topic in order to structure them
- Each of the contributors is responsible for their part of the work, as specified in the table

> [!NOTE]
> This review table will be updated, so it is not a final version

| Topic | Title | Year | Authors | Paper | Code | Summary |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| Datasets with simultaneous fMRI-EEG signals <br> [@kisnikser](https://github.com/kisnikser) | An open-access dataset of naturalistic viewing using simultaneous EEG-fMRI | 2023 | Qawi K. Telesford et al. | [scientific data](https://www.nature.com/articles/s41597-023-02458-8) | [GitHub](https://github.com/NathanKlineInstitute/NATVIEW_EEGFMRI) | TODO |
|      | Open multimodal iEEG-fMRI dataset from naturalistic stimulation with a short audiovisual film | 2022 | Julia Berezutskaya et al. | [scientific data](https://www.nature.com/articles/s41597-022-01173-0) | [GitHub #1](https://github.com/UMCU-RIBS/ieeg-fmri-dataset-validation), [GitHub #2](https://github.com/UMCU-RIBS/ieeg-fmri-dataset-quickstart) |      |
|      | Simultaneous EEG and functional MRI data during rest and sleep from humans | 2023 | Yameng Gu et al. | [Data in Brief](https://www.sciencedirect.com/science/article/pii/S2352340923001774) | [Download](https://openneuro.org/datasets/ds003768/versions/1.0.11) |      |
|      | Simultaneous and independent electroencephalography and magnetic resonance imaging: A multimodal neuroimaging dataset | 2023 | Jonathan Gallego-Rudolf et al. | [Data in Brief](https://www.sciencedirect.com/science/article/pii/S2352340923007461) | [Download](https://data.mendeley.com/datasets/crhybxpdy6/2) |     |   
| Methods using fMRI <br> [@DorinDaniil](https://github.com/DorinDaniil) | Natural scene reconstruction from fMRI signals using generative latent diffusion | 2023 | Furkan Ozcelik et al. | [arXiv](https://arxiv.org/abs/2303.05334) | [GitHub](https://github.com/ozcelikfu/brain-diffuser) | Use sklearn ridge regression as an fMRI encoder |
|      | fMRI-based Decoding of Visual Information from Human Brain Activity: A Brief Review | 2021 | Shuo Huang et al. | [Springer Link](https://link.springer.com/article/10.1007/s11633-020-1263-y) | - | Analyze architectures |
|      | Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity | 2023 | Zijiao Chen et al. | [arXiv](https://arxiv.org/abs/2305.11675) | [GitHub](https://github.com/jqin4749/MindVideo), [Website](https://www.mind-video.com/) | TODO |
|      | High-resolution image reconstruction with latent diffusion models from human brain activity | 2023 | Yu Takagi et al. | [arXiv](https://arxiv.org/abs/2306.11536) | [GitHub](https://github.com/yu-takagi/StableDiffusionReconstruction) | |
| Methods using EEG <br> [@sem-k32](https://github.com/sem-k32) | Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion | 2024 | Dongyang Li et al. | [arXiv](https://arxiv.org/abs/2403.07721v5) | [GitHub](https://github.com/dongyangli-del/EEG_Image_decode)  | EEG encoder = Transformer -> CNN (for spatiotemp. dependencies) -> MLP; EEG context vector is used to reconstruct image CLIP-vector. The latter is used in diffusion model to gen images |
|      | NeuroGAN: image reconstruction from EEG signals via an attention-based GAN | 2022 | Rahul Mishra et al. | [Springer Link](https://link.springer.com/article/10.1007/s00521-022-08178-1) | - | CNN encoder for EEG incorporated into GAN's generator. $$ Loss = Loss_{\text{GAN}} + Loss_{\text{image classification}} + Loss_{\text{perceptial loss}} $$ |
|      | EEG2IMAGE: Image Reconstruction from EEG Brain Signals | 2023 | Prajwal Singh et al. | [arXiv](https://arxiv.org/abs/2302.10121) | [GitHub](https://github.com/prajwalsingh/EEG2Image) | Individual EEG feature extractor (constastive learning) + conditioned GAN for image generation |
|      | Image Reconstruction from Electroencephalography Using Latent Diffusion | 2024 | Teng Fei et al. | [arXiv](https://arxiv.org/abs/2404.01250) | [GitHub](https://github.com/desa-lab/EEG-Image-Reconstruction) |   info-gypsy   |
| SOTA fMRI encoders <br> [@DorinDaniil](https://github.com/DorinDaniil) |     |     |     |     |     |     |
|      |      |      |      |      |      |      |
| SOTA EEG encoders <br> [@sem-k32](https://github.com/sem-k32) |     |     |     |     |     |     |
|      |      |      |      |      |      |      |
| SOTA methods for image generation <br> [@kisnikser](https://github.com/kisnikser) |     |     |     |     |     |     |
|      |      |      |      |      |      |      |
