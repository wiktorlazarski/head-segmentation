import typing as t

from torchvision import transforms

IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]


def create_preprocessing_pipeline(input_image_size: t.Tuple[int, int]) -> t.Callable:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(input_image_size),
            transforms.Normalize(mean=IMAGENET_MEANS, std=IMAGENET_STDS),
        ]
    )
