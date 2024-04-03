## Leap Labs Programming Challenge.

When run directly, `main.py` will display the panda image unperturbed, then display an image that has been adversarially perturbed using PGD towards the 0 (tench) class.

I created this code originally in a jupyter session, since it was much easier to plot and modify things without re-loading, before copying in here.
Initially I tried the gradient sign method, but it wasn't effective on the EfficientNet model.

This code ran with
```
torch=2.2.1
torchvision=0.17.1
```
