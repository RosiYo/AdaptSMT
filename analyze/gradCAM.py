

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F


class GradCAM:
    # Static variables to store gradients and activations
    gradients = None
    activations = None

    @staticmethod
    def save_gradients(module, grad_in, grad_out):
        """Save gradients from the backward pass."""
        GradCAM.gradients = grad_out[0]

    @staticmethod
    def save_activations(module, input, output):
        """Save activations from the forward pass."""
        GradCAM.activations = output

    @staticmethod
    def calculate_grad_cam(model, input_tensor, target_layer, target_class):
        """
        Calculate Grad-CAM heatmaps for sequence prediction.

        Args:
            model: The model to analyze.
            input_tensor: Input image tensor of shape [1, 3, H, W].
            target_layer: The target layer for Grad-CAM.
            target_class: Target class index for which to calculate Grad-CAM.

        Returns:
            text_sequence: List of predicted tokens as text.
            heatmaps: List of Grad-CAM heatmaps for each decoding step.
        """
        # Register hooks once
        target_layer.register_forward_hook(GradCAM.save_activations)
        target_layer.register_backward_hook(GradCAM.save_gradients)

        # Store predicted sequence and heatmaps
        text_sequence = []
        heatmaps = []

        # Initial sequence with <bos>
        predicted_sequence = torch.tensor(
            [[model.w2i['<bos>']]], device=input_tensor.device
        )

        # Generate sequence step by step
        for _ in range(model.maxlen - predicted_sequence.shape[-1]):
            model.zero_grad()

            # Forward pass through the encoder (constant for all decoding steps)
            encoder_output = model.forward_encoder(input_tensor)

            # Forward pass through the decoder
            predictions = model.forward_decoder(encoder_output, predicted_sequence)
            logits = predictions.logits[:, :, -1]  # Latest logits
            predicted_token = torch.argmax(logits, dim=1).item()

            # Append predicted token to the sequence
            predicted_sequence = torch.cat(
                [predicted_sequence, torch.argmax(logits, dim=1, keepdim=True)], dim=1
            )

            # Stop if <eos> is predicted
            predicted_token = str(predicted_token)
            if model.i2w[predicted_token] == '<eos>':
                break
            text_sequence.append(model.i2w[predicted_token])

            # Backward pass for Grad-CAM
            logits[0, target_class].backward()

            # Compute weights directly for 1D activations
            weights = GradCAM.gradients.squeeze(-1)  # Remove last singleton dimension
            grad_cam = (weights * GradCAM.activations.squeeze(-1)).sum(dim=1)  # Sum over features
            grad_cam = torch.relu(grad_cam)  # Apply ReLU

            # Normalize Grad-CAM
            grad_cam = grad_cam - grad_cam.min()
            grad_cam = grad_cam / (grad_cam.max() + 1e-8)

            # Append heatmap
            heatmaps.append(grad_cam.detach().cpu().numpy())

        return text_sequence, heatmaps

    @staticmethod
    def generate_heatmaps(
        model,
        dataset,
        token,
        target_layer,
        folder="analyze/heatmaps",
        nsamples=10,
        fps=10
    ):
        """
        Generate Grad-CAM heatmaps for each sample in a dataset and store as individual videos.

        Args:
            model: The model to analyze.
            dataset: Dataset to analyze.
            token: Identifier for filtering samples.
            target_layer: The target layer for Grad-CAM.
            folder: Base folder to store the heatmaps and videos.
            nsamples: Number of samples to process.
            fps: Frames per second for the output videos.

        Returns:
            None
        """
        token_idx = model.w2i[token]
        os.makedirs(folder, exist_ok=True)

        for i, (img, _) in enumerate(dataset):
            if i == nsamples:
                break

            # Create folder for this sample
            sample_folder = os.path.join(folder, f"sample_{i}")
            os.makedirs(sample_folder, exist_ok=True)

            # Generate Grad-CAM and text sequence
            text_sequence, heatmaps = GradCAM.calculate_grad_cam(
                model, img, target_layer, token_idx
            )

            # Determine video properties
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_np = (img_np * 255).astype(np.uint8)
            height, width, _ = img_np.shape

            video_path = os.path.join(sample_folder, f"sample_{i}_grad_cam.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            print(f"Processing sample {i}: Saving heatmaps and video to {sample_folder}")

            # Generate heatmaps for each decoding step
            for step, grad_cam in enumerate(heatmaps):
                # Resize Grad-CAM heatmap to match image size
                grad_cam_resized = cv2.resize(grad_cam, (width, height))
                heatmap = (grad_cam_resized * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map

                # Overlay heatmap on the original image
                overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

                # Add the predicted text sequence to the overlay
                overlay_with_text = overlay.copy()
                cv2.putText(
                    overlay_with_text,
                    f"Step {step}: {' '.join(text_sequence)}",
                    (10, 30),  # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # Font scale
                    (255, 255, 255),  # Font color (white)
                    2,  # Thickness
                    cv2.LINE_AA
                )

                # Save overlay image and write to video
                heatmap_path = os.path.join(sample_folder, f"heatmap_step_{step}.png")
                cv2.imwrite(heatmap_path, overlay_with_text)
                video_writer.write(overlay_with_text)

            video_writer.release()
            print(f"Video saved for sample {i} at {video_path}")
