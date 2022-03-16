from __future__ import annotations

import typing as t

import segmentation_models_pytorch as smp
import torch


class HeadSegmentationModel(smp.Unet):
    @staticmethod
    def load_from_checkpoint(ckpt_path: str) -> HeadSegmentationModel:
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))

        hparams = ckpt["hyper_parameters"]
        neural_net = HeadSegmentationModel(
            encoder_name=hparams["encoder_name"],
            encoder_depth=hparams["encoder_depth"],
            pretrained=False,
            nn_image_input_resolution=hparams["nn_image_input_resolution"],
        )

        weigths = {
            k.replace("neural_net.", ""): v for k, v in ckpt["state_dict"].items()
        }
        neural_net.load_state_dict(weigths, strict=False)

        return neural_net

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
