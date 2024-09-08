# Wet Voice Detection
In collaboration with Sara Albert and The Laryngology Innovation Lab led by Dr. Anaïs Rameau.  

Machine Learning techniques to accurately predict swallowing difficulties in patients by analyzing the alterations in their voice and speech patterns before and after they swallow water.

This project involves the collection and meticulous processing of voice and speech data from a diverse cohort of patients who have already undergone objective swallow evaluations through standard clinical procedures. This private dataset is used to train a supervised ML model. By utilizing various ML algorithms and feature extraction techniques, the model is developed to identify and quantify subtle changes in speech that are indicative of swallowing difficulties. This could potentially aid in early diagnosis and enhance therapeutic strategies. Ultimately improving the quality of diagnosis for those affected by such conditions.

## Data Description

We collected a dataset of 90 audio files before the swallowing and 90 after. 
The labels are binary (aspiration absent, or normal - 0, aspiration present - 1)

Class distribution is 34 aspiration and 56 normal.

The average length  - 13 seconds
The maximum length - 36 seconds
The minimum length - 1 second

## Methodologies
We evaluated several strategies: 


1. Advanced features are created by calculating statistical properties (mean and std) on top of the spectral, chroma, and temporal features.
For example: [mean_centroid, std_centroid, mean_rolloff, std_rolloff, mean_bandwidth, std_bandwidth, mean_chroma, std_chroma, mean_zcr, std_zcr]
We test traditional ML methods such as KNN classifier, SVM and random forest trees.
We also utilize T-SNE, as it is highly effective at revealing clusters or groups within data. It can separate data points clearly, making it easier to identify patterns that are not obvious in higher dimensions.
Due to its sensitivity to local structure, t-SNE can help identify outliers or anomalies in the data.


2. CNN (MobileNetV2) on Pre-Post Pairs Stacked
Each audio file can be represented as a melspectrogram.
Two pairs of melspectrograms are provided as an input to the MobileNetV2 model. Trained with transfer learning from the pretrained weights initialization.  

3. CNN (MobileNetV2) on Pre-Post Pairs Stacked with Augmentations
For augmentation strategy we chose to use gaussian noise, time stretching, pitch shift, shift, and for all of them probability to be applied is 50%.
We use https://iver56.github.io/audiomentations/ library for the augmentations.

4. CNN (MobileNetV2) Results on Pre-Post Pairs Stacked - Cross-validation
Here, we use a stratified k-fold cross-validation technique.
Stratified cross-validation is particularly useful and important for handling datasets where class distribution is imbalanced—that is, where some classes are significantly underrepresented compared to others.
This setup will help perform a more statistically robust cross-validation, leading to better generalized models, especially beneficial in varied real-world scenarios.

   
## Learnings

* CNN on melspectrograms performs better than traditional ML approaches (KNN, Random Forest, SVM) trained on engineered features from the audio signal.
* Augmentations helped decrease the overfit and slightly improved the performance. 
* This dataset may be too small to achieve meaningful results with deep learning. The metrics are changing significantly depending on the split. The severe overfit is a sign that the task can be solvable, but it can't generalize well when being trained only on 72 cases. 

 
