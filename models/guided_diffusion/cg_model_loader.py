import torch
from models.guided_diffusion.unet import UNetModel, EncoderUNetModel

NUM_CLASSES = 1000

def create_model(
    image_size=256,  # Fixed for ADM (e.g., ImageNet256)
    num_channels=256,
    num_res_blocks=2,
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    dropout=0.0,
    channel_mult="",
    class_cond=True,
    use_checkpoint=False,
    use_scale_shift_norm=True,
    resblock_updown=True,
    use_fp16=True,
    learn_sigma=True,
):
    if channel_mult == "":
        channel_mult = (1, 1, 2, 2, 4, 4)  # Default for 256x256
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = [image_size // int(res) for res in attention_resolutions.split(",")]

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=6 if learn_sigma else 3,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=NUM_CLASSES,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
    )

def create_classifier(
    image_size=256,
    classifier_use_fp16=True,
    classifier_width=128,
    classifier_depth=2,
    classifier_attention_resolutions="32,16,8",
    classifier_use_scale_shift_norm=True,
    classifier_resblock_updown=True,
    classifier_pool="attention",
):
    channel_mult = (1, 1, 2, 2, 4, 4)  # Default for 256x256
    attention_ds = [image_size // int(res) for res in classifier_attention_resolutions.split(",")]

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=NUM_CLASSES,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

def load_cg_model(model_path, classifier_path):
    model = create_model()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    if model.use_fp16:
        model.convert_to_fp16()

    classifier = create_classifier()
    state_dict = torch.load(classifier_path, map_location='cpu')
    classifier.load_state_dict(state_dict)
    if classifier.use_fp16:
        classifier.convert_to_fp16()

    return model, classifier