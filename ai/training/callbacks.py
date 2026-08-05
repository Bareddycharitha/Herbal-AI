import os
import torch


class ModelCheckpoint:

    def __init__(self, checkpoint_dir):

        self.best_accuracy = 0.0

        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        accuracy
    ):

        if accuracy > self.best_accuracy:

            self.best_accuracy = accuracy

            checkpoint = {

                "epoch": epoch,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "scheduler_state_dict": scheduler.state_dict(),

                "best_accuracy": accuracy

            }

            torch.save(
                checkpoint,
                os.path.join(
                    self.checkpoint_dir,
                    "best_model.pth"
                )
            )

            print("\n✅ Best model saved.")