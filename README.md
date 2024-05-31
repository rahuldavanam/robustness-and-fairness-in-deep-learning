# Robustness and fairness in deep learning

This code is mostly based on Dongbin Na's (https://github.com/ndb796) code with a few modifications to suit the requirements of my task. 

### To see the progress so far:
* Run `basic_training.py` to conventionally train the model with benign data.
* Run `pgd_adversarial_training.py` to perform PGD training of the model with benign and adversarial data. This code is with an untargeted attack approach.
* Run `basic_training_test.py` and `pgd_adversarial_training_test.py` once you're done training with `basic_training.py` and `pgd_adversarial_trainng.py` respectively to obtain the benign and adversarial accuracy.
* In addition to this the confusion matrix for benign and adversarial cases was plotted to analyse the classwise distribution of misclassifications when untargeted attacks.

### Future steps:
* Perform PGD training with targeted adversarial training using adversarial examples with all target classes except ground truth for each sample.
* Repeat targeted PGD training considering different norms of robust losses across classes instead of averaging.
* Analyze class-wise natural and robust accuracies in each scenario.

Further progress will be pushed to the repository soon when it's done.
