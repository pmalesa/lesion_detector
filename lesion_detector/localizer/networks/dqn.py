import tensorflow as tf
from keras import Model, layers
from keras.applications.resnet50 import ResNet50


class DQN(Model):
    """
    Implementation of a simple DQN model, that extracts features
    via pretrained CNN into a 1D vector and contatenates it with the
    normalized bounding box parameters (x, y, w, h), which serves
    as input and outputs Q-values for action_dim discrete actions
    """

    def __init__(self, pretrained=False, action_dim=9, image_shape=(512, 512, 3)):
        super().__init__()

        self._action_dim = action_dim
        self._image_shape = image_shape

        base_model = ResNet50(include_top=False, input_shape=image_shape, pooling="avg")

        self._cnn = base_model
        self._cnn_out_dim = 2048

        # MLP component for bounding box parameters
        self._bbox_mlp = tf.keras.Sequential(
            [layers.Dense(32, activation="relu"), layers.Dense(32, activation="relu")]
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

    def call(self, inputs, training=False):
        """
        Method being a single forward pass through the network.
        The training parameter can be used to specify different
        behaviour during training and inference.
        inputs argument is a tuple (img_tensor, bbox_coords), where:
        - img_tensor has shape (batch, 512, 512, 3)
        - bbox_coords has shape (batch, 4)
        """

        img_tensor, bbox_coords = inputs

        img_features = self._cnn(img_tensor, training=training)

        bbox_features = self._bbox_mlp(bbox_coords, training=training)

        combined = tf.concat([img_features, bbox_features], axis=-1)

        q_values = self._comb_mlp(combined, training=training)

        return q_values
