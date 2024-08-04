# Robustness and fairness in deep learning

This code is inspired by Dongbin Na's (https://github.com/ndb796) code and also uses his customized ResNet-18 model and his implementation of basic and untargeted training.

### To see the different phases of the project:
* Run `basic_training.py` to conventionally train the model with benign data.
* Run `pgd_adversarial_training.py` to perform PGD-untargeted training of the model with benign and adversarial data. 
* Run `pgd_targeted_adversarial_training.py` to perform PGD-targeted training of the model with benign and adversarial data. The adversarial data here is created such that an equal number of images are taken from each class. Then each image is misclassified into all the target labels except the true label. A custom DataLoader called `BalancedBatchSampler` has been created for this purpose
* Run `basic_training_test.py`, `pgd_adversarial_training_test.py` and `pgd_targeted_adversarial_training_test.py` once you're done training with the respective codes to obtain the benign, untargeted and targeted attack data. 
* In addition to this the confusion matrix for benign and adversarial cases was plotted to analyse the classwise distribution of misclassifications and accuracies on each scenario.
* Run all the codes starting with `loss_norms _ targeted/untargeted _ l2/square_root.py` to perform untargeted / targeted training on l2 / l0.5 norms. Run their respective `test.py` codes to obtain the accuracies and losses on benign, untargeted and targeted attack data. Confusion matrices are also implemented and saved to analyze class-wise accuracies for each scenario.


### Observations:
* Adversarial training has definitely aided in predicting all types of image data, with a much better accuracy for adversarial examples compared to basic training.
* Even if targeted training has a better training accuracy compared to untargeted training, the latter has a better test accuracy.
* Untargeted training has shown a greater degree of robustness with untargeted and targeted attacks, while on the other targeted training has not given adequate accuracy for untargeted attacks.
* The class with the best accuracy is the automobile (untargeted training = 72% | targeted training = 85%) which makes it the most robust class.
* The class with the worst accuracy is the cat (untargeted training = 23% | targeted training = 53%) which makes it more vulnerable to misclassifications. But the accuracy could be improved by implementing “balanced” targeted training.
* Both the training methods learned better when square root loss norms was implemented as they have a better training accuracy.
* The discrepancy in class-wise accuracies with untargeted training on untargeted attacks is very high (accuracy range = 49%), while on the other hand, the range is fairly lower for targeted attacks (accuracy range = 33%) with the average accuracy having increased too.
* For both, untargeted and targeted attacks, we can observe that there is high robustness when trained on average losses rather than l2 or l.
