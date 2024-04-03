import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F


with open("imagenet-simple-labels.json", "r") as f:
    IMAGENET_CLASSES = json.load(f)


def load_efficientnet():
    """
    Load pretrained and frozen VGG16 in eval mode, and its preprocesser.

    Returns:
        Tuple of the pytorch VGG16 model and the image preprocessing
        function for this model.
    """
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # Load model and freeze.
    model = torchvision.models.efficientnet_b0(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model, preprocess


def load_image(image_path, preprocess):
    """
    Load an image from disk and process into a tensor for input to model.

    Args:
        image_path: path to the image file.
        preprocess: the torchvision preprocessing function for the model.

    Returns:
        A normalised and preprocessed image tensor.
    """
    image = Image.open(image_path)
    image = preprocess(image)
    return image.unsqueeze(0)   # Add a batch dim.


def imshow(image_tensor, norm_mean, norm_std, title=""):
    """
    Display a tensor as an image, reversing normalisation.

    Args:
        image_tensor: the normalised image tensor to display.
        norm_mean: mean that was used to normalise the image.
        norm_std: standard deviation that was used to normalise the image.
        title: title for the figure.
    """
    image_tensor =image_tensor.squeeze(0)              # Remove batch dimension if present
    image_tensor =image_tensor.detach().cpu().numpy()  # Convert to numpy array
    image_tensor =image_tensor.transpose((1, 2, 0))    # Reorder dims to HWC (for Pytorch)

    # Denormalize
    image_tensor = norm_std * image_tensor + norm_mean
    image_tensor = image_tensor.clip(0, 1)

    plt.imshow(image_tensor)
    plt.axis("off")
    plt.title(title)
    plt.show()


def decode(model, image):
    """
    Get the predicted class and probability from the model.

    Args:
        model: the torchvision model from which predictions come.
        image: the image tensor to predict on.

    Returns:
        A tuple of the predicted class and its probability.
    """
    with torch.no_grad():
        output = model(image)
        class_probs = F.softmax(output, dim=1).squeeze(0)
    prediction = torch.argmax(class_probs)
    return prediction, class_probs[prediction]


def decode_and_display(model, image, preprocess):
    """
    Open matplotlib figure of image with model prediction and confidence.

    Args:
        model: the torchvision model from which predictions come.
        image: the image tensor to predict on.
        preprocess: the torchvision preprocessing function for the model.
    """
    cls, conf = decode(model, image)
    fig = plt.figure()
    title = f"Class: {IMAGENET_CLASSES[cls]} ({cls}). Confidence: {conf * 100:.2f}%."
    imshow(image, preprocess.mean, preprocess.std, title)


def targeted_pgd_attack(
        model,
        original_image,
        target_class,
        epsilon=0.05,
        step_size=0.01,
        num_iter=40,
        ):
    """
    Perform a targeted Projected Gradient Descent attack.
    Based on paper: Towards Deep Learning Models Resistant to Adversarial Attacks,
    Madry et al. (2019).

    Args:
        model: pytorch model being used.
        original_image: original image, transformed/normalised for model.
        target_class: integer label of class being targeted.
        epsilon: l_infinity maximum difference of perturbed image from original.
        step_size: step size for the gradient iterations.
        num_iter: number of iterations of projected gradient descent performed.

    Returns:
        A perturbed (normalised for model input) image targeting target_class.
    """
    # Clone the original image.
    perturbed_img = original_image.clone().detach().requires_grad_(True)

    target_class = torch.tensor([target_class])

    for _ in range(num_iter):
        loss = F.cross_entropy(model(perturbed_img), target_class)
        loss.backward()
        perturbed_img = perturbed_img - step_size * perturbed_img.grad.data.sign()

        # Project l infinity norm back.
        delta = torch.clamp(perturbed_img - original_image, min=-epsilon, max=epsilon)
        perturbed_img = original_image + delta

        # Don't let gradients accumulate through multiple steps.
        perturbed_img.detach_().requires_grad_(True)

    return perturbed_img



if __name__ == "__main__":
    # Example usage for the functions above.
    model, preprocess = load_efficientnet()
    image = load_image("panda.jpg", preprocess)
    target_class = 0
    perturbed_image = targeted_pgd_attack(model, image, target_class)

    decode_and_display(model, image, preprocess)
    decode_and_display(model, perturbed_image, preprocess)
