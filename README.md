## Visual stimuli reconstruction from simultaneous fMRI-EEG signals

[![website](https://img.shields.io/badge/Project-Page-orange)](https://intsystems.github.io/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/)

[![License](https://badgen.net/github/license/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG?color=green)](https://github.com/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG)](https://github.com/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG.svg?color=0088ff)](https://github.com/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG.svg?color=7f29d6)](https://github.com/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/pulls)

<table>
    <tr>
        <td align="left"> <b> Authors </b> </td>
        <td> Daniil Dorin, Nikita Kiselev, Ernest Nasyrov, Kirill Semkin </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Vadim Strijov, DSc </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Andrey Grabovoy, PhD </td>
    </tr>
</table>

## 🔎 Overview
![scheme](https://github.com/user-attachments/assets/2777eab1-de35-4c4c-8309-6030bf2892ee)
- Reconstruct images, which participants viewed during the simultaneous fMRI-EEG procedure
- There is no method that uses a simultaneous fMRI-EEG signal (only fMRI / EEG separately)
- [LinkReview](https://github.com/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/blob/main/linkreview.md)

## 💡 Abstract
How to decode human vision through neural signals has attracted a long-standing interest in neuroscience and machine learning. Modern contrastive learning and generative models improved the performance of visual decoding and reconstruction based on functional Magnetic Resonance Imaging (fMRI) and electroencephalography (EEG). However, combining these two types of information is difficult to decode visual stimuli, including due to a lack of training data. In this study, we present an end-to-end fMRI-EEG based visual reconstruction zero-shot framework, consisting of multiple tailored brain encoders and fuse module, which projects neural signals from different sources into the shared subspace as the CLIP embedding, and a two-stage multi-pipe fMRI-EEG-to-image generation strategy. In stage one, fMRI and EEG are embedded to align the high-level CLIP embedding, and then the prior diffusion model refines combined embedding into image priors. In stage two, we input this combined embedding to a pre-trained diffusion model. The experimental results indicate that our fMRI-EEG-based visual zero-shot framework achieves SOTA performance in reconstruction, highlighting the portability, low cost, and hight temporal and spatial resolution of combined fMRI-EEG, enabling a wide range of BCI applications.

## 🧰 Method
![method scheme](https://github.com/user-attachments/assets/77254da9-8c26-4255-b289-ab6621f0f832)

## 🔗 Useful links
- [Brain Imaging Data Structure (BIDS) Starter Kit](https://bids-standard.github.io/bids-starter-kit/index.html)
- [OpenNeuroDatasets](https://github.com/OpenNeuroDatasets)
