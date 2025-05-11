import tensorflow as tf
from keras import Model, layers
from keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="localizer.networks")
class DQN(Model):
    """
    Implementation of a simple DQN model, that extracts features
    via pretrained CNN into a 1D vector, which serves as input to
    an MLP head and then outputs Q-values for action_dim discrete actions
    """

    def __init__(self, action_dim=9, image_shape=(128, 128, 3), **kwargs):
        super().__init__(**kwargs)

        self._action_dim = action_dim
        self._image_shape = image_shape

        self._cnn = VGG16(include_top=False, input_shape=image_shape, pooling="avg")
        for layer in self._cnn.layers:
            layer.trainable = False

        # Unfreeze the last block of the VGG16 network (3 last conv layers)
        self._cnn.layers[-3].trainable = True
        self._cnn.layers[-4].trainable = True
        self._cnn.layers[-5].trainable = True
        self._cnn_out_dim = 2048

        # MLP head
        self._mlp_head = tf.keras.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(self._action_dim, activation=None),
            ]
        )

    def build(self, input_shape=None):
        """
        Method called by Keras before the first forward pass
        """
        super().build(input_shape)

    def call(self, inputs):
        """
        Method being a single forward pass through the network.
        inputs argument is a 2D array of cropped patch pixel data
        of shape (batch, patch_data)
        """

        img_patch_tensor = inputs

        # training=True means that there will be updates made on
        # batch norm layers if needed. It will not freeze the unfreezed layers!
        img_features = self._cnn(img_patch_tensor, training=False)

        q_values = self._mlp_head(img_features, training=True)

        return q_values

    def get_config(self):
        """
        Method that returns the config dictionary for serialization.
        """

        config = super().get_config()
        config.update(
            {
                "action_dim": self._action_dim,
                "image_shape": self._image_shape,
            }
        )

        return config
