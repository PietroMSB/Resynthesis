# Resynthesis

This repository is intended to help carry out resynthesis experiments as described in the following papers: 
Bongini, Pietro, et al. "Training-free Source Attribution of AI-generated Images via Resynthesis." 17th IEEE INTERNATIONAL WORKSHOP ON. INFORMATION FORENSICS AND SECURITY. (WIFS) (2025).

To use this repository, you first need to download the following dataset from Kaggle: https://www.kaggle.com/datasets/pietrob92/resynthesis-dataset
Once downloaded, put the dataset inside the main repo directory, under name "Dataset".

The files allow to build a basic experimental setup by extracting features and then using them to attribute through resynthesis-based methodologies.
- parameters_and_functions.py contains utility functions, classes, and parameters to carry out the experiments
- clip_images.py contains the code to extract clip features from the resynthesized images
- clip_references.py contains the code to extract clip features from the original images
- select_by_distance.py contains the code to apply distance selection to the previously extracted features

# Cite

If you use this work, please cite the following publications:

Bongini, Pietro, et al. "Training-free Source Attribution of AI-generated Images via Resynthesis." 17th IEEE INTERNATIONAL WORKSHOP ON. INFORMATION FORENSICS AND SECURITY. (WIFS) (2025).

@inproceedings{bongini2025training,
  title={Training-free Source Attribution of AI-generated Images via Resynthesis},
  author={Bongini, Pietro and Molinari, Valentina and Costanzo, Andrea and Tondi, Benedetta and Barni, Mauro},
  booktitle={17th IEEE INTERNATIONAL WORKSHOP ON. INFORMATION FORENSICS AND SECURITY. (WIFS)},
  year={2025}
}
