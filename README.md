# Robustness and fairness in deep learning

This code is mostly based on Dongbin Na's (https://github.com/ndb796) code with a few modifications to suit the requirements of my task. 

### To see the different phases of the project:
* Run `basic_training.py` to conventionally train the model with benign data.
* Run `pgd_adversarial_training.py` to perform PGD-untargeted training of the model with benign and adversarial data. 
* Run `pgd_targeted_adversarial_training.py` to perform PGD-targeted training of the model with benign and adversarial data. The adversarial data here is created such that an equal number of images are taken from each class. Then each image is misclassified into all the target labels except the true label. A custom DataLoader called `BalancedBatchSampler` has been created for this purpose
* Run `basic_training_test.py`, `pgd_adversarial_training_test.py` and `pgd_targeted_adversarial_training_test.py` once you're done training with the respective codes to obtain the benign, untargeted and targeted attack data. 
* In addition to this the confusion matrix for benign and adversarial cases was plotted to analyse the classwise distribution of misclassifications and accuracies on each scenario.
* Run all the codes starting with `loss_norms _ targeted/untargeted _ l2/square_root.py` to perform untargeted / targeted training on l2 / l0.5 norms. Run their respective `test.py` codes to obtain the accuracies and losses on benign, untargeted and targeted attack data. Confusion matrices are also implemented and saved to analyze class-wise accuracies for each scenario.
