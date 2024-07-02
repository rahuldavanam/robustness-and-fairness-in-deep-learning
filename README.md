# Robustness and fairness in deep learning

This code is mostly based on Dongbin Na's (https://github.com/ndb796) code with a few modifications to suit the requirements of my task. 

### To see the progress so far:
* Run `basic_training.py` to conventionally train the model with benign data.
* Run `pgd_adversarial_training.py` to perform PGD-untargeted training of the model with benign and adversarial data. 
* Run `pgd_targeted_adversarial_training.py` to perform PGD-targeted training of the model with benign and adversarial data. The adversarial data here is created such that an equal number of images are taken from each class and then each image is misclassified into all the target labels except the true label. A custom DataLoader called `BalancedBatchSampler` has been created for this purpose
* Run `basic_training_test.py` and `pgd_adversarial_training_test.py` and `pgd_targeted_adversarial_training_test.py` once you're done training with the respective codes to obtain the benign and adversarial accuracy.
* In addition to this the confusion matrix for benign and adversarial cases was plotted to analyse the classwise distribution of misclassifications when untargeted attacks.

### Future steps:
* Repeat targeted PGD training considering different norms of robust losses across classes instead of averaging.
* Analyze class-wise natural and robust accuracies in each scenario.

Further progress will be pushed to the repository soon when it's done.
