import torch.nn as nn
import torch.nn.functional as F
from realesrgan.loss_util_gfpp import _vgg19, _vgg_face
from realesrgan.loss_util_gfpp import apply_vggface_normalization
from realesrgan.loss_util_gfpp import apply_imagenet_normalization
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class PerceptualLossGFPP(nn.Module):
    def __init__(
        self,
        layers_weight={"relu_1_1": 0.03125, "relu_2_1": 0.0625, "relu_3_1": 0.125, "relu_4_1": 0.25, "relu_5_1": 1.0},
        n_scale=3,
        vgg19_loss_weight=1.0,
        vggface_loss_weight=1.0,
        include_style_loss=True,
    ):
        super().__init__()
        self.vgg19 = _vgg19(layers_weight.keys())  # Assuming _vgg19 is a function to initialize VGG-19
        self.vggface = _vgg_face(layers_weight.keys())  # Assuming _vgg_face is a function to initialize VGG-Face
        self.criterion = nn.L1Loss()
        self.mse_criterion = nn.MSELoss()
        self.layers_weight = layers_weight
        self.n_scale = n_scale
        self.vgg19_loss_weight = vgg19_loss_weight
        self.vggface_loss_weight = vggface_loss_weight
        self.include_style_loss = include_style_loss
        self.vgg19.eval()
        self.vggface.eval()
        print("Perceptual Loss GFPP initialized")

    def _gram_matrix(self, features):
        (b, ch, h, w) = features.size()
        features = features.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def forward(self, input, target, face_coords_xywh):
        # print("input shape:", input.shape)
        # print("target shape:", target.shape)
        # if input.shape[-1] != 512:
        #     input = F.interpolate(input, mode="bilinear", size=(512, 512), align_corners=False)
        #     target = F.interpolate(target, mode="bilinear", size=(512, 512), align_corners=False)

        self.vgg19.eval()
        self.vggface.eval()
        loss = 0

        ### Old Version -> Extract VGG-Face features without cropping
        # features_vggface_input = self.vggface(apply_vggface_normalization(input))
        # features_vggface_target = self.vggface(apply_vggface_normalization(target))

        ### New Version -> Extract VGG-Face features with cropping using face_coords_xywh
        x, y, w, h = face_coords_xywh
        input_face = input[:, :, y:y+h, x:x+w]
        target_face = target[:, :, y:y+h, x:x+w]
        features_vggface_input = self.vggface(apply_vggface_normalization(input_face))
        features_vggface_target = self.vggface(apply_vggface_normalization(target_face))
        # print("VGG-Face Features:")
        # for key, feat in features_vggface_input.items():
        #     print(key, feat.shape)


        # Extract VGG-19 features
        input = apply_imagenet_normalization(input)
        target = apply_imagenet_normalization(target)
        features_vgg19_input = self.vgg19(input)
        features_vgg19_target = self.vgg19(target)
        # print("VGG-19 Features:")
        # for key, feat in features_vgg19_input.items():
        #     print(key, feat.shape)

        for layer, weight in self.layers_weight.items():
            # VGG-Face Perceptual Loss
            loss += self.vggface_loss_weight * weight * self.criterion(
                features_vggface_input[layer], features_vggface_target[layer].detach()) / 255

            # VGG-19 Perceptual Loss
            loss += self.vgg19_loss_weight * weight * self.criterion(
                features_vgg19_input[layer], features_vgg19_target[layer].detach())

            # Style Loss (using VGG-19)
            if self.include_style_loss:
                gram_input = self._gram_matrix(features_vgg19_input[layer])
                gram_target = self._gram_matrix(features_vgg19_target[layer])
                loss += weight * self.mse_criterion(gram_input, gram_target.detach())

        # Multi-scale Perceptual Loss
        for i in range(self.n_scale):
            input = F.interpolate(input, mode="bilinear", scale_factor=0.5, align_corners=False)
            target = F.interpolate(target, mode="bilinear", scale_factor=0.5, align_corners=False)
            features_vgg19_input = self.vgg19(input)
            features_vgg19_target = self.vgg19(target)
            loss += weight * self.criterion(features_vgg19_input[layer], features_vgg19_target[layer].detach())

        return loss, None
