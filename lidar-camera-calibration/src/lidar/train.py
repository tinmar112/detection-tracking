import torch
import torch.nn.functional as F
from torch import nn


def detection_loss(cls_pred, reg_pred, yaw_pred, cls_target, reg_target, yaw_target):
    """
    Batch size B
    
    cls_pred:  [B, 1, H, W] -> P(car)
    reg_pred:  [B, 6, H, W]
    yaw_pred:  [B, 2, H, W]

    cls_target: [B, 1, H, W]
    reg_target: [B, 6, H, W]
    yaw_target: [B, 2, H, W]

    Positive cells are cls_target == 1.
    """

    cls_loss = F.binary_cross_entropy_with_logits(cls_pred, cls_target.float())

    # only calculate box/yaw losses for positive cells
    positive_mask = cls_target.bool()

    if positive_mask.any():

        # [B, 1, H, W] -> [B, H, W]
        pos_mask = positive_mask.squeeze(1)

        # [B, 6, H, W] -> [number_positive, 6]
        reg_pred_pos = reg_pred.permute(0, 2, 3, 1)[pos_mask]
        reg_target_pos = reg_target.permute(0, 2, 3, 1)[pos_mask]

        # [B, 2, H, W] -> [number_positive, 2]
        yaw_pred_pos = yaw_pred.permute(0, 2, 3, 1)[pos_mask]
        yaw_target_pos = yaw_target.permute(0, 2, 3, 1)[pos_mask]

        reg_loss = F.smooth_l1_loss(reg_pred_pos, reg_target_pos)

        yaw_loss = F.smooth_l1_loss(yaw_pred_pos, yaw_target_pos)

    else:
        # Keep gradients connected to the model
        reg_loss = reg_pred.sum() * 0.0
        yaw_loss = yaw_pred.sum() * 0.0

    total_loss = cls_loss + reg_loss + yaw_loss

    return total_loss, {"total": total_loss.detach(),
                        "cls": cls_loss.detach(),
                        "reg": reg_loss.detach(),
                        "yaw": yaw_loss.detach()}

def train(model: nn.Module,
          device: torch.device,
          num_epochs: int,
          train_loader) -> None:

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3,
                                  weight_decay=1e-4)

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            pillars = batch["pillars"].to(device)
            pillar_coords = batch["pillar_coords"].to(device)

            cls_target = batch["cls_target"].to(device)
            reg_target = batch["reg_target"].to(device)
            yaw_target = batch["yaw_target"].to(device)

            cls_pred, reg_pred, yaw_pred = model(pillars, pillar_coords)

            loss, loss_dict = detection_loss(cls_pred, reg_pred, yaw_pred, cls_target, reg_target, yaw_target)

            optimizer.zero_grad()
            loss.backward()

            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()

        mean_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, loss: {mean_loss}")
