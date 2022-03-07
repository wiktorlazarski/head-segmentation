import typing as t

import segmentation_models_pytorch as smp


class HeadSegmentationModel(smp.Unet):
    def __init__(
        self,
        encoder_name: str,
        encoder_depth: int,
        pretrained: bool,
        nn_image_input_resolution: int,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights="imagenet" if pretrained else None,
            decoder_use_batchnorm=True,
            decoder_channels=self._decoder_channels(
                nn_image_input_resolution, encoder_depth
            ),
            decoder_attention_type=None,
            in_channels=3,
            classes=2,
        )

    def _decoder_channels(
        self, nn_image_input_resolution: int, encoder_depth: int
    ) -> t.Tuple[int]:
        return tuple(
            [nn_image_input_resolution // (2 ** i) for i in range(encoder_depth)]
        )
