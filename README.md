# Investigating Model Generalization Through Regularization, Pretraining, and Dataset Diversity

## Overview

Machine learning models must learn patterns that generalize beyond their training data. However, a model can achieve high training accuracy without learning representations that transfer reliably to unseen examples. This is particularly challenging when the training dataset is limited in size or visual diversity.

This project investigates how **regularization, model architecture, and pretrained visual representations influence classification performance and generalization**. Using two image datasets, DBI and SDD, I conducted a series of experiments to examine the effects of dropout, training from scratch versus using pretrained weights, and evaluating models on a dataset with greater visual diversity.

The experiments explore three related questions:

* **Regularization:** Does dropout help a CNN generalize by reducing overfitting?
* **Model capacity and pretraining:** How does a deeper residual network trained from scratch compare with a simpler CNN, and how much can pretrained visual representations improve performance?
* **Cross-dataset generalization:** Do models trained on DBI retain their performance when evaluated on the more visually diverse SDD dataset?

The goal is to understand not only which models perform better, but also what their training behavior and cross-dataset performance suggest about their ability to generalize.

## Background

### Regularization and Dropout

Regularization techniques help control overfitting by discouraging models from relying excessively on patterns specific to the training data.

Dropout is a regularization method that randomly sets a fraction of neuron activations to zero during training. This encourages the network to learn representations that are less dependent on particular activations and can improve generalization.

To investigate its effect, two CNNs with the same architecture were trained on DBI: one with dropout and one without. Their training, validation, and test accuracies were compared to assess whether dropout improved performance on unseen examples.

### Model Capacity and Residual Networks

More complex architectures can represent more intricate patterns, but their additional capacity does not automatically translate into better generalization. When training data is limited, a model may fit the training examples without learning features that transfer reliably to unseen data.

ResNet18 introduces residual connections that facilitate the training of deeper neural networks. To investigate the relationship between model capacity and generalization, a ResNet18 trained from scratch was compared with the custom CNN.

Additional ResNet architectures were then evaluated using pretrained weights to investigate whether previously learned visual representations could improve classification performance.

### Transfer Learning and Pretrained Representations

Models pretrained on large image datasets can learn reusable visual representations, including features related to edges, textures, shapes, and object structure.

These representations may provide a useful starting point when the target dataset is smaller or less diverse than the pretraining dataset.

In this project, pretrained ResNet18, ResNet34, ResNet50, and ResNeXt50 models were adapted to the seven-class classification task. Their original parameters were frozen, and a new classification head was trained on DBI.

This approach is more specifically described as frozen-feature transfer learning, or feature extraction, rather than full fine-tuning.

### Dataset Diversity and Generalization

A model's performance on a held-out test set does not necessarily indicate how well it will perform on data with different visual characteristics.

Differences in object positioning, occlusion, image clarity, and other visual conditions can create a distribution shift between training and evaluation data.

To investigate this, the trained ResNet models were evaluated on SDD, a dataset with greater visual diversity than DBI. Comparing performance across the two datasets provides an indication of how well the learned representations transfer to a different visual distribution.

## Experimental Setup

### Datasets

Two image datasets were used:

| Dataset | Purpose                                               |
| ------- | ----------------------------------------------------- |
| DBI     | Training, validation, and in-dataset testing          |
| SDD     | Cross-dataset evaluation of the trained ResNet models |

Both datasets were loaded using the project's `DataManager`, which creates training, validation, and test data loaders.

The experiments use seven output classes. The SDD evaluation is intended to examine performance under greater visual diversity rather than to train the models on SDD.

### Models

Seven model configurations were investigated.

| Model      | Initialization        | Training configuration              |
| ---------- | --------------------- | ----------------------------------- |
| Simple CNN | Random initialization | With dropout                        |
| Simple CNN | Random initialization | Without dropout                     |
| ResNet18   | Random initialization | Trained from scratch                |
| ResNet18   | ImageNet pretrained   | Frozen backbone, trained classifier |
| ResNet34   | ImageNet pretrained   | Frozen backbone, trained classifier |
| ResNet50   | ImageNet pretrained   | Frozen backbone, trained classifier |
| ResNeXt50  | ImageNet pretrained   | Frozen backbone, trained classifier |

The custom CNN consists of four convolutional layers followed by a classifier. The ResNet-based experiments compare residual network architectures with different depths and feature-extraction designs.

For the pretrained models, the original fully connected layer was replaced with a new classification head for the seven target classes. The pretrained backbone parameters were frozen during training.

### Training and Evaluation

Models were trained on DBI, with training and validation accuracy recorded over ten epochs. The final test accuracy was used to compare performance on held-out DBI examples.

The CNN experiments used a learning rate of 0.001, while the ResNet experiments used a learning rate of 0.0003. The configured batch size was 128.

After training, the ResNet models were evaluated on SDD using its test split. The resulting accuracies were compared with their DBI test accuracies to examine cross-dataset generalization.

The experiments were implemented in Python using PyTorch and Torchvision. Training, dataset loading, and evaluation were organized into reusable components in the `src/` directory.

## Results and Analysis

### 1. Dropout: CNN With vs. Without Regularization

| Metric                    | CNN with dropout | CNN without dropout |
| ------------------------- | ---------------: | ------------------: |
| Final training accuracy   |              41% |                 37% |
| Final validation accuracy |              31% |                 21% |
| Test accuracy             |              35% |                 32% |

The CNN trained with dropout achieved a test accuracy of 35%, compared with 32% for the model without dropout. It also achieved higher final validation accuracy.

The training curves provide additional context. The model without dropout developed a larger gap between training and validation accuracy, suggesting that improvements on the training set were not translating as effectively to unseen validation examples.

Dropout likely helped by discouraging excessive dependence on specific neuron activations during training. This interpretation is consistent with the observed improvement in validation and test performance.

Although the improvement was modest, the results support the hypothesis that regularization can help a model generalize better on a dataset with limited visual diversity.

### 2. Simple CNN vs. ResNet18 Trained From Scratch

| Metric                    | CNN with dropout | ResNet18 from scratch |
| ------------------------- | ---------------: | --------------------: |
| Final training accuracy   |              41% |                   76% |
| Final validation accuracy |              31% |                   26% |
| Test accuracy             |              35% |                   41% |

ResNet18 achieved higher test accuracy than the custom CNN, but its training behavior suggests a greater tendency to overfit.

Its training accuracy increased to 76%, while validation accuracy finished at 26%. This substantial gap indicates that the model learned patterns that improved its fit to the training data without corresponding improvements in validation performance.

One possible explanation is that ResNet18 has greater representational capacity than the custom CNN. With limited training data, a more complex model may learn dataset-specific patterns that do not transfer as reliably to unseen examples.

However, the test accuracy results show that the relationship between overfitting and generalization is not straightforward. Despite its larger train-validation gap, ResNet18 achieved 41% test accuracy, compared with 35% for the CNN.

These results suggest that a model's training fit, validation behavior, and test performance should be considered together. They also motivate investigating whether pretrained representations can make a larger architecture more effective on a limited target dataset.

### 3. Pretrained ResNet Models

| Model     | DBI test accuracy |
| --------- | ----------------: |
| ResNet18  |               80% |
| ResNet34  |               84% |
| ResNet50  |               93% |
| ResNeXt50 |               93% |

The pretrained ResNet models substantially outperformed ResNet18 trained from scratch, which achieved 41% test accuracy.

This improvement suggests that pretrained visual representations provided a valuable advantage. Rather than learning all visual features from the target dataset, the pretrained models could reuse features learned during large-scale image pretraining and adapt a new classifier to DBI.

The results also show a performance difference between the pretrained architectures. ResNet50 and ResNeXt50 achieved the highest test accuracy at 93%, followed by ResNet34 at 84% and ResNet18 at 80%.

The stronger performance of ResNet50 and ResNeXt50 is consistent with the possibility that their architectures and pretrained representations provide more useful features for this classification task. However, the experiments do not isolate the effect of model depth from other architectural differences.

Overall, these results suggest that pretraining was more consequential than simply increasing model capacity without pretraining.

### 4. Generalization to the More Visually Diverse SDD Dataset

The trained ResNet models were evaluated on SDD to investigate whether their performance transferred to a dataset with greater visual diversity.

| Model                  | DBI accuracy | SDD accuracy |    Change (SDD − DBI) |
| ---------------------- | -----------: | -----------: | --------------------: |
| ResNet18, from scratch |          41% |          28% | -13 percentage points |
| ResNet18, pretrained   |          80% |          81% |   +1 percentage point |
| ResNet34, pretrained   |          84% |          80% |  -4 percentage points |
| ResNet50, pretrained   |          93% |          86% |  -7 percentage points |
| ResNeXt50, pretrained  |          93% |          90% |  -3 percentage points |

The model trained from scratch experienced the largest decrease in accuracy, falling from 41% on DBI to 28% on SDD. This is consistent with its weaker validation behavior during training and suggests that the features it learned did not transfer as effectively to the second dataset.

In contrast, the pretrained models retained substantially higher accuracy on SDD, ranging from 80% to 90%. Their cross-dataset performance was relatively close to their DBI performance, despite the increased visual diversity.

These results support the hypothesis that pretrained representations can improve robustness to changes in visual conditions. Pretraining may expose models to a broader range of visual patterns, enabling them to reuse features that remain useful when the target dataset changes.

ResNet18 even achieved one percentage point higher accuracy on SDD than on DBI. This small difference should not be interpreted as evidence that SDD is inherently easier: sampling variability, dataset composition, and differences in the evaluation examples may all contribute.

Importantly, the experiment evaluates the combined effect of pretraining and architecture. It does not directly measure the diversity of the pretraining data or isolate which visual characteristics account for the improved transfer.

## Key Findings

The experiments highlight several relationships between model design and generalization:

1. **Regularization can improve generalization.** The CNN trained with dropout achieved higher validation and test accuracy than its counterpart without dropout, consistent with reduced overfitting.

2. **Greater model capacity does not guarantee better generalization.** ResNet18 trained from scratch developed a substantial train-validation gap, showing that a more complex architecture can struggle when learning from limited data.

3. **Pretrained representations can provide a substantial advantage.** All four pretrained ResNet configurations outperformed ResNet18 trained from scratch on DBI, suggesting that reusable visual features were valuable for the target task.

4. **In-dataset performance does not fully characterize generalization.** Evaluating the models on SDD revealed a much larger accuracy decrease for ResNet18 trained from scratch than for the pretrained models.

5. **Architecture and pretraining should be considered together.** The highest recorded DBI accuracy came from ResNet50 and ResNeXt50 with pretrained weights, rather than from the model trained from scratch with the greatest training accuracy.

## Limitations and Future Work

These experiments provide an initial comparison rather than a definitive measurement of generalization.

Several extensions could strengthen the findings:

* **Repeat experiments across random seeds:** Report mean accuracy and standard deviation to assess whether the observed differences are consistent across runs.
* **Evaluate more regularization techniques:** Compare dropout with weight decay, data augmentation, and other approaches to controlling overfitting.
* **Compare frozen features with full fine-tuning:** Unfreeze some or all pretrained backbone layers to determine whether adapting the pretrained representations improves performance.
* **Control for architecture and initialization:** Compare pretrained and randomly initialized models using the same architecture and training procedure to better isolate the contribution of pretraining.
* **Investigate dataset composition:** Examine class distributions, image characteristics, and possible differences in preprocessing between DBI and SDD.
* **Analyze per-class performance:** Use confusion matrices and class-wise metrics to identify which categories are most affected by visual variation.
* **Measure robustness under controlled variation:** Evaluate performance under specific changes in positioning, occlusion, and image clarity rather than relying solely on aggregate cross-dataset accuracy.

## Repository Structure

```text
dataset-diversity-generalization/
├── data/
│   ├── DBIsubset.zip
│   └── SDDsubset.zip
├── src/
│   ├── data_manager.py
│   ├── simple_cnn.py
│   └── trainer.py
└── model_performance.ipynb
```

## Running the Experiments

The experiments were developed in Google Colab.

1. Clone the repository.
2. Extract the DBI and SDD dataset archives.
3. Install the required Python dependencies, including PyTorch and Torchvision, along with the other dependencies used by the project.
4. Open `model_performance.ipynb` in Google Colab or a compatible Jupyter environment.
5. Run the notebook cells to load the datasets, initialize the models, train the configurations, and evaluate their performance.

The notebook contains the experiment definitions, training and evaluation calls, accuracy plots, and cross-dataset comparisons.

## Conclusion

This project investigates how regularization, architecture, and pretrained visual representations relate to generalization in image classification.

The results suggest that dropout can help reduce overfitting in a simple CNN, while training a higher-capacity network from scratch may be insufficient when the target dataset has limited diversity. Pretrained ResNet models achieved substantially higher accuracy and retained better performance on a second, more visually diverse dataset.

Taken together, the experiments highlight an important distinction: a model's ability to fit its training data is not the same as its ability to generalize. Effective regularization and reusable pretrained representations can help bridge that gap, particularly when training data does not capture the full range of visual conditions encountered during evaluation.
