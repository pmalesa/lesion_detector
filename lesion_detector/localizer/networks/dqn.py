import tensorflow as tf
from keras import Model, layers
from keras.applications.vgg16 import VGG16


class DQN(Model):
    """
    Implementation of a simple DQN model, that extracts features
    via pretrained CNN into a 1D vector and contatenates it with the
    normalized bounding box parameters (x, y, w, h), which serves
    as input and outputs Q-values for action_dim discrete actions
    """

    def __init__(self, action_dim=9, image_shape=(128, 128, 3)):
        super().__init__()

        self._action_dim = action_dim
        self._image_shape = image_shape

        self._cnn = VGG16(include_top=False, input_shape=image_shape, pooling="avg")
        for layer in self._cnn.layers:
            layer.trainable = False

        # Unfreeze the last trainale Conv layer at position -3
        self._cnn.layers[-3].trainable = True
        self._cnn_out_dim = 2048

        # # MLP component for bounding box parameters
        # self._bbox_mlp = tf.keras.Sequential(
        #     [layers.Dense(32, activation="relu"), layers.Dense(32, activation="relu")]
        # )

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

    # TODO
    def call(self, inputs):
        """
        Method being a single forward pass through the network.
        The training parameter can be used to specify different
        behaviour during training and inference.
        inputs argument is a 2D array of cropped patch pixel data
        of shape (batch, patch_data)
        """

        img_patch_tensor = inputs

        # TODO check if the training parameter is necessary
        img_features = self._cnn(img_patch_tensor, training=False)

        # Add history of 10 previous actions ...
        # combined = tf.concat([img_features, bbox_features], axis=-1)

        q_values = self._comb_mlp(img_features, training=True)

        return q_values
