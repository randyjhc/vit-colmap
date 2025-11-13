"""
Simple script to print layer information for BEiT, DINOv2, and DINOv3 models from Hugging Face.
Uses PyTorch's built-in APIs for model inspection.
"""

from transformers import (
    BeitModel,
    Dinov2Model,
)


def print_model_info(model, model_name):
    """Print model architecture using PyTorch's built-in methods."""
    print(f"\n{'='*80}")
    print(f"{model_name} Model Architecture")
    print(f"{'='*80}\n")

    # Use PyTorch's built-in print
    print(model)

    print(f"\n{'Parameter Summary':-^80}\n")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")


def main():
    print("Loading models from Hugging Face...")

    # # BEiT Model
    print("\n[1/3] Loading BEiT model...")
    beit_model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224")
    print_model_info(beit_model, "BEiT (microsoft/beit-base-patch16-224)")

    # DINOv2 Model
    print("\n[2/3] Loading DINOv2 model...")
    dinov2_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    print_model_info(dinov2_model, "DINOv2 (facebook/dinov2-base)")

    # DINOv3 Model
    # print("\n[3/3] Loading DINOv3 model...")
    # try:
    #     dinov3_model = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
    #     print_model_info(dinov3_model, "DINOv3 (facebook/dinov3-vit7b16-pretrain-lvd1689m)")
    # except Exception as e:
    #     print(f"\nNote: DINOv3 model not available: {e}")
    #     print("DINOv3 might not be released yet or uses a different identifier.\n")

    # print("="*80)
    # print("Done!")
    # print("="*80)


if __name__ == "__main__":
    main()
