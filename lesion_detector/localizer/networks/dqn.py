import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from keras import Model, layers
from keras.applications.vgg16 import VGG16

@register_keras_serializable(package="localizer.networks")
class DQN(Model):
    """
    Implementation of a simple DQN model, that extracts features
    via pretrained CNN into a 1D vector and contatenates it with the
    normalized bounding box parameters (x, y, w, h), which serves
    as input and outputs Q-values for action_dim discrete actions
    """

    def __init__(self, action_dim=9, image_shape=(64, 64, 3)):
        super().__init__()

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

        # # MLP component for previous actons
        self._hist_mlp = tf.keras.Sequential(
            [layers.Dense(64, activation="relu"), layers.Dense(32, activation="relu")]
        )

        # Combined MLP (Q-network)
        self._comb_mlp = tf.keras.Sequential(
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
        and previous actions of shape (batch, patch_data, prev_actions)
        """

        img_patch_tensor, prev_actions_tensor = inputs

        # training=True means that there will be updates made on
        # batch norm layers if needed. It will not freeze the unfreezed layers!
        img_features = self._cnn(img_patch_tensor, training=False)

        history_features = self._hist_mlp(prev_actions_tensor, training=True)

        combined = tf.concat([img_features, history_features], axis=-1)

        q_values = self._comb_mlp(combined, training=True)

        return q_values
    
    def get_config(self):
        """
        Method that returns the config dictionary for serialization.
        """

        config = super().get_config()
        config.update({
            "action_dim": self._action_dim,
            "image_shape": self._image_shape,
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Method called by Keras to create a new instance of a class
        from given config.
        """

        config.pop("trainable", None)
        config.pop("dtype", None)

        return cls(**config)