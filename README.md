# EgoCast

**[Project page](https://bcv-uniandes.github.io/egocast-wp/) &bull;
[arXiv](https://arxiv.org/abs/2412.02903)**

<table>
    <tr>
        <td>
            Maria Escobar<sup>1</sup>, Juanita Puentes<sup>1</sup>, Cristhian Forigua<sup>1</sup>, Jordi Pont-Tuset<sup>2</sup>, Kevis-Kokitsi Maninis<sup>2</sup>, Pablo Arbelaez<sup>1</sup>.
            <strong>EgoCast: Forecasting Egocentric Human Pose in the Wild.</strong>
            arXiv, 2025.
        </td>
    </tr>
</table>
<sup>1</sup><em>Universidad de Los Andes</em>, <sup>2</sup><em>Google DeepMind</em>



---
## Overview

EgoCast is a novel framework for full-body pose forecasting. We use visual and proprioceptive cues to accurately predict body motion.

---
## Getting started

1. **Clone the repository.**
   ```bash
   git clone https://github.com/BCV-Uniandes/EgoCast.git
   ```
2. **Install general dependencies.**

   To set up the environment and install the necessary dependencies, run the following commands:
   ```bash
   cd EgoCast
   conda create -n egocast python=3.11 -y
   conda activate egocast
   pip install .
   ```

3. **Download model checkpoint.**

      We use the [EgoVPL model](https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7/view) from [EgoVPL implementation](https://github.com/showlab/EgoVLP). Please download and put the checkpoint under `model_zoo/`
      
## Dataset & Preparation

We utilize [EgoExo-4D](https://ego-exo4d-data.org/), a large-scale, multi-modal, multi-view video dataset collected across 13 cities worldwide. This dataset serves as a benchmark for egocentric and exocentric human motion analysis.

   For training, our model leverages camera poses and egocentric video data. 

  1. **Data Download**
    
      To download the dataset, follow the instructions provided in the [EgoExo-4D documentation](https://docs.ego-exo4d-data.org/).
    
      To obtain metadata and body pose annotations, run the following command:
    
      ```bash
       egoexo -o dataset --parts annotations --benchmarks egopose --release v2
      ```
    
      To download the downscaled takes (448p resolution) of the egocentric videos, run the following command:
    
      ```bash
       egoexo -o dataset --parts annotations --benchmarks egopose --release v2
      ```
2. **Data Preparation**

   To train our model, the downloaded egocentric video takes must be converted into individual frames. This step extracts frames from the videos and saves them as images for further processing.

   ```bash
   python video2image.py
   ```

## Current-Frame Estimation Module

The Current-Frame Estimation Module predicts the full-body pose at the current timestamp using camera poses and, optionally, egocentric video. This eliminates the reliance on ground-truth body poses at test time, enabling real-world applicability. We offer two training approaches:

### Training

1. **IMU-Based Approach** (Uses only camera poses)
    Train using only IMU (headset pose) data:

    ```bash
   python main_train_egocast.py -opt options/train_egocast_imu.json
   ```

2. **EgoCast Approach** (Uses camera poses and egocentric video)
    Train using both camera pose and visual data:

    ```bash
   python main_train_egocast.py -opt options/train_egocast_video.json
   ```

### Test

1. **IMU-Based Testing** (Uses only camera poses)
    Run the following command to evaluate the IMU-based model:

    ```bash
   python main_test_egocast.py -opt options/test_egocast_imu.json
   ```

2. **EgoCast Testing** (Uses camera poses and egocentric video)
    Run the following command to test the model using both IMU data and video:

    ```bash
   python main_test_multiprocessing.py -opt options/test_egocast_multiprocessing.json
   ```
## Forecasting Module



## Citations

If you find EgoCast useful for your work please cite:

```
@article{escobar2025egocast,
  author    = {Escobar, Maria and Puentes, Juanita and Forigua, Cristhian and Pont-Tuset, Jordi and Maninis, Kevis-Kokitsi and Arbeláez, Pablo},
  title     = {EgoCast: Forecasting Egocentric Human Pose in the Wild},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      = {2025},
}
```
## License and Acknowledgement

This project borrows heavily from [AvatarPoser]([https://github.com/openai/guided-diffusion](https://github.com/eth-siplab/AvatarPoser)), we thank the authors for their contributions to the community.<br>


## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
