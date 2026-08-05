import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:

    def __init__(self, model, target_layer=None):

        self.model = model
        self.gradients = None
        self.activations = None

        # Use provided target layer or find default
        if target_layer is not None:
            self.target_layer = target_layer
        else:
            # Default for tf_efficientnetv2_s
            self.target_layer = model.conv_head

        self.target_layer.register_forward_hook(
            self.save_activation
        )

        self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, class_index):

        self.model.zero_grad()

        output = self.model(image_tensor)

        score = output[:, class_index]

        score.backward()

        gradients = self.gradients.detach()

        activations = self.activations.detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze()

        cam -= cam.min()

        cam /= (cam.max() + 1e-8)

        cam = cam.cpu().numpy()

        return cam


def overlay_heatmap(original_image, cam):

    h, w = original_image.shape[:2]

    cam = cv2.resize(cam, (w, h))

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original_image,
        0.5,
        heatmap,
        0.5,
        0
    )

    return overlay