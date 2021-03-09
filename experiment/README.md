## Code

This directory contains the source code of the brain2pix models with fixed and learnable receptive fields.

Both folder contains the following files: 
- util.py: Devices are defined here, which is needed to run the mxnet framework on GPU or CPU.
- modules.py: This file contains all the functions used for loading and combining the data to make the input for the model such as get_signals_from_run(), get_RFs(), and get_inputsROI.  The file also contains teh functions that were used to make the iterator for the training of synthetic data and real data (make_synthetic_iterator() and make_iterator() respectively).
- discriminator.py: The discriminator which is used to discriminate between real and fake images. The loss used for this model is Binary Cross Entropy loss, which is also defined in this file.

Files in "fixedRF/":
- generator.py: The generator is defined here, it makes of the discriminator (defined in distriminator.py) as a loss. This specific generator is only used for the "No vgg" ablation experiment.
- generator_vgg.py: This is the generator containing vgg loss component. This generator is used for the main model.
- train_model.py: This is the training loop for the fixed RF model.

Files in "learnableRF/":
- generator_learnRF.py: The generator that makes use of learnable receptive field layers.
- learn_RF.py: Training loop for the learnableRF model

---

## Additional details of experiments

#### Baseline models:
Additional details of the baseline models can be found below:

As mentioned in the paper, the first (simple) baseline reconstructed the stimuli by inverting a linear Gaussian encoding model1 with MAP estimation [36]. The second (more complex) baseline reconstructed the stimuli by maximizing the likelihood of a nonlinear-linear encoding model with a SqueezeNet component [18] as the nonlinear feature extractor (second max-pooling layer outputs of the SqueezeNet V2 architecture pretrained on ImageNet). All densities were assumed to be Gaussian except for the prior which was203an empirical natural image prior constructed from the training set [30, 31].

Specifically for the complex baseline:
For each frame, the SqueezeNet model was used to extract 128 x 11 x 11 dimensional stimulus features. 
For each voxel, a Gaussian likelihood (mean and diagonal covariance) of voxel responses given stimulus pixels was estimated on the training set with ridge regression. 
For each 0.7 s of the test set, the top three training frames resulting in the highest likelihoods of test V1, V2 and V3 responses in that 0.7 s given the training frame were averaged to obtain the reconstruction of the test frame in that 0.7 s.

Specifically for the simple baseline:

A Gaussian prior (zero mean & #pixels x #pixels covariance) was estimated on the training stimulus pixels.
For each voxel, a Gaussian likelihood (mean and diagonal covariance) of voxel responses given stimulus pixels was estimated on the training set with ridge regression.
A Gaussian posterior (mean and covariance) of stimulus pixels given voxel responses was obtained with the Bayes'rule. For each 0.7 s of the test set, the test frame was reconstructed from the V1, V2 and V3 responses in that 0.7 s with MAP estimation on the posterior.

#### ROI experiment:
For the ROI experiments, the fixedRF model was used. The differences between the codes in running the ROI experiment and the one in the fixedRF map is that in the ROI experiment, the signals of only one brain region is used to train at a time. The experiment for one brain region only, V1 for instance, can be done by changing the following in the codes:


Defining the iterators are done by the following:
``` python
train_iter = modules.make_iterator_preprocessed('training','V1','V2', 'V3', batch_size=batch_size, shuffle=True)
test_iter = modules.make_iterator_preprocessed('testing', 'V1','V2', 'V3', batch_size=batch_size, shuffle=True)
```
For V1 only, iterator needs to become this:
``` python
train_iter = modules.make_iterator_preprocessed('training','V1', batch_size=batch_size, shuffle=True)
test_iter = modules.make_iterator_preprocessed('testing', 'V1', batch_size=batch_size, shuffle=True)
```


and then during the training loop:

``` python
    for RFsignalsV1,RFsignalsV2,RFsignalsV3, targets in tqdm(train_iter, total = len(train_iter)):
        inputs1 = modules.get_inputsROI(RFsignalsV1, RFlocs_V1_overlapped_avg, context)
        inputs2 = modules.get_inputsROI(RFsignalsV2, RFlocs_V2_overlapped_avg, context)
        inputs3 = modules.get_inputsROI(RFsignalsV3, RFlocs_V3_overlapped_avg, context)
        inputs = concat(inputs1, inputs2, inputs3, dim=1)
```

needs to become:

``` python
    for RFsignalsV1, targets in tqdm(train_iter, total = len(train_iter)):
        inputs = modules.get_inputsROI(RFsignalsV1, RFlocs_V1_overlapped_avg, context)
```

This ensures that ONLY the inputs of the V1 region in the brain is used to train the model.


#### Ablation experiment:

``` python    
def __call__(self, p: float, p_hat: NDArray, y: NDArray, y_hat: NDArray) -> NDArray:
        
        dis_loss = self._alpha * mean(self._bce(p_hat, full(p_hat.shape, p))) 
        
        gen_loss_vgg = self._beta_vgg * mean(self._vgg(y_hat, y))
        gen_loss_pix = self._beta_pix * mean(self._l1(y_hat, y))
        
        return dis_loss + gen_loss_vgg + gen_loss_pix
```
        

The above loss function was used for the final brain2pix model. For the ablation studies, the fixedRF model was used to run without specific loss components:

- "no feature" experiment: We made use of the above defined loss function but <i>without</i> the`gen_loss_vgg` component.
- "no adversarial" experiment": We made use of the same loss function defined above but <i>without</i> the `dis_loss` component. In this case, we only trained the generator.


#### Baseline experiments:

- dcgan_baseline: reimplemented the [Shen  et. al.](https://www.frontiersin.org/articles/10.3389/fncom.2019.00021/full) model on the Dr. Who dataset
- 6 framed output: changed the last layer to output 6 channels.
- split data: Experiment with different amount of training frames (20%, 40%, 60%, 100%)
- FC layers: Replaced all convolution layers with fully connected layers

