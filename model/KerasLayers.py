import tensorflow as tf
import numpy as np
from lifelines.utils import concordance_index


class Embed(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension, trainable=False, triangular=False):
        super(Embed, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.trainable = trainable
        self.triangular = triangular
        self.embedding_matrix = None
        self.embedding_matrix_padded = None

    def build(self, input_shape):
        if self.triangular:
            self.embedding_matrix = self.add_weight(shape=[self.embedding_dimension, self.embedding_dimension], initializer=tf.constant_initializer(value=np.tri(self.embedding_dimension)), trainable=self.trainable, dtype=tf.float32)
        else:
            self.embedding_matrix = self.add_weight(shape=[self.embedding_dimension, self.embedding_dimension], initializer=tf.keras.initializers.identity(), trainable=self.trainable, dtype=tf.float32)
        self.embedding_matrix_padded = tf.concat([tf.zeros([1, self.embedding_dimension]), self.embedding_matrix], axis=0)

    def call(self, inputs, **kwargs):
        return tf.gather(self.embedding_matrix_padded, inputs, axis=0)

class Activations:
    class ASU(tf.keras.layers.Layer):
        def __init__(self, trainable=True, lower_asymptote=0., upper_asymptote=1., alpha_init=1., bias_init=None):
            super(Activations.ASU, self).__init__()
            self.trainable = trainable
            self.lower_asymptote = lower_asymptote
            self.upper_asymptote = upper_asymptote
            self.alpha_init = alpha_init
            self.lower_alpha = None
            self.upper_alpha = None
            self.bias_init = bias_init
            self.bias = None

        @staticmethod
        def activation_function(x, lower_asymptote, upper_asymptote, lower_alpha, upper_alpha):
            x_2 = x ** 2
            lower_sqrt = (lower_alpha + x_2) ** (1 / 2)
            upper_sqrt = (upper_alpha + x_2) ** (1 / 2)
            return lower_asymptote + ((upper_asymptote - lower_asymptote) * ((x + lower_sqrt) / (lower_sqrt + upper_sqrt)))

        def build(self, input_shape):
            self.lower_alpha = self.add_weight(shape=[input_shape[-1], ],
                                               initializer=tf.keras.initializers.constant(self.alpha_init),
                                               dtype=tf.float32, trainable=self.trainable)
            self.upper_alpha = self.add_weight(shape=[input_shape[-1], ],
                                               initializer=tf.keras.initializers.constant(self.alpha_init),
                                               dtype=tf.float32, trainable=self.trainable)
            if self.bias_init is not None:
                self.bias = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.alpha_init), dtype=tf.float32, trainable=self.trainable)


        def call(self, inputs, **kwargs):
            return self.activation_function(inputs + self.bias if self.bias is not None else inputs,
                                            lower_asymptote=self.lower_asymptote, upper_asymptote=self.upper_asymptote,
                                            lower_alpha=tf.exp(self.lower_alpha), upper_alpha=tf.exp(self.upper_alpha))

    class ARU(tf.keras.layers.Layer):
        def __init__(self, trainable=True, alpha_init=0., bias_init=None):
            super(Activations.ARU, self).__init__()
            self.trainable = trainable
            self.alpha_init = alpha_init
            self.alpha = None
            self.bias_init = bias_init
            self.bias = None

        @staticmethod
        def activation_function(x, alpha):
            return (x + ((alpha + (x ** 2)) ** (1 / 2))) / 2

        def build(self, input_shape):
            self.alpha = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.alpha_init), dtype=tf.float32, trainable=self.trainable)
            if self.bias_init is not None:
                self.bias = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.bias_init), dtype=tf.float32, trainable=True)

        def call(self, inputs, **kwargs):
            return self.activation_function(inputs + self.bias if self.bias is not None else inputs, alpha=tf.exp(self.alpha))


class StrandWeight(tf.keras.layers.Layer):
    def __init__(self, n_features, trainable=True, strand_init=0.):
        super(StrandWeight, self).__init__()
        self.n_features = n_features
        self.trainable = trainable
        self.strand_init = strand_init
        self.strand_weight = None

    def build(self, input_shape):
        self.strand_weight = self.add_weight(shape=[self.n_features, ], initializer=tf.keras.initializers.constant(self.strand_init), dtype=tf.float32, trainable=self.trainable)

    def call(self, inputs, **kwargs):
        return (Activations.ASU.activation_function(self.strand_weight, lower_asymptote=0., upper_asymptote=1., lower_alpha=1., upper_alpha=1.)[tf.newaxis, tf.newaxis, ...] * (inputs[..., tf.newaxis] - 1)) + 1


class Dense:
    class Gate(tf.keras.layers.Layer):
        def __init__(self, units, activation, bias_kwargs):
            super(Dense.Gate, self).__init__()
            self.units, self.activation, self.bias_kwargs = units, activation, bias_kwargs

        def build(self, input_shape):
            self.bias = self.add_weight(shape=(self.units, ), dtype=tf.float32, **self.bias_kwargs)

        def call(self, inputs, **kwargs):
            return self.activation(inputs + self.bias)


class Ragged:
    class MapFlatValues(tf.keras.layers.Layer):
        def __init__(self, op):
            super(Ragged.MapFlatValues, self).__init__()
            self._supports_ragged_inputs = True
            self.op = op

        def call(self, inputs, **kwargs):
            return tf.ragged.map_flat_values(self.op, inputs)

    class Dense(tf.keras.layers.Layer):
        def __init__(self, units, activation=None):
            super(Ragged.Dense, self).__init__()
            self._supports_ragged_inputs = True
            self.units, self.activation = units, activation
            self.ragged_layer, self.tensor_layer, self.activation_layer = None, None, None

        def build(self, inputs):
            self.ragged_layer = tf.keras.layers.Dense(units=self.units, activation=None, use_bias=False)
            self.tensor_layer = tf.keras.layers.Dense(units=self.units, activation=None, use_bias=False)
            self.activation_layer = Dense.Gate(units=self.units, activation=self.activation, bias_kwargs=dict(initializer=tf.keras.initializers.constant(0), trainable=True))

        def call(self, inputs, **kwargs):
            ragged_dot = tf.ragged.map_flat_values(self.ragged_layer, inputs[0]) + tf.expand_dims(self.tensor_layer(inputs[1]), inputs[0].ragged_rank)
            return tf.ragged.map_flat_values(self.activation_layer, ragged_dot)

    class Attention(tf.keras.layers.Layer):
        def __init__(self, pooling='sum', regularization=.2, layers=[16, ]):
            super(Ragged.Attention, self).__init__()
            self.pooling = pooling
            self._supports_ragged_inputs = True
            self.layers = layers
            self.regularization = regularization
            self.attention_layers = []
            for i in layers:
                self.attention_layers.append(tf.keras.layers.Dense(units=i, activation='relu'))
            self.attention_layers.append(tf.keras.layers.Dense(units=1, activation=Activations.ASU(), activity_regularizer=tf.keras.regularizers.l1(regularization)))

        def call(self, inputs, **kwargs):
            if self.pooling == 'dynamic':
                attention_weights = [inputs[1]]
                for i in self.attention_layers:
                    attention_weights.append(tf.ragged.map_flat_values(i, attention_weights[-1]))
                attention_weights = attention_weights[-1]
                pooled = tf.reduce_sum(tf.ragged.map_flat_values(tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
                                                                 [tf.ragged.map_flat_values(tf.expand_dims, attention_weights, axis=2),
                                                                  tf.ragged.map_flat_values(tf.expand_dims, inputs[0], axis=1)]), axis=1)

            else:
                attention_weights = [inputs]
                for i in self.attention_layers:
                    attention_weights.append(tf.ragged.map_flat_values(i, attention_weights[-1]))
                attention_weights = attention_weights[-1]
                if self.pooling == 'mean':
                    pooled = tf.reduce_sum(tf.ragged.map_flat_values(tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
                                                                     [tf.ragged.map_flat_values(tf.expand_dims, attention_weights, axis=2),
                                                                      tf.ragged.map_flat_values(tf.expand_dims, inputs, axis=1)]), axis=1)
                    pooled = pooled / tf.expand_dims(tf.reduce_sum(attention_weights, axis=1) + .000000001, axis=-1)
                else:
                    pooled = tf.reduce_sum(tf.ragged.map_flat_values(tf.keras.layers.Lambda(lambda x: x[0] * x[1]),
                                                                        [tf.ragged.map_flat_values(tf.expand_dims, attention_weights, axis=2),
                                                                         tf.ragged.map_flat_values(tf.expand_dims, inputs, axis=1)]), axis=1)

            return pooled, attention_weights


class Losses:         
    class cox_loss(tf.keras.losses.Loss):
        def __init__(self, name='coxl'):
            super(Losses.cox_loss, self).__init__(name=name)

        def call(self, y_true, y_pred):
            tf.print(y_pred)
            time_value = tf.squeeze(y_true[:, 0])
            event = tf.squeeze(y_true[:, 1])
            event = tf.cast(event, dtype=tf.bool)
            score = tf.squeeze(y_pred)
            ix = tf.where(event)[:, 0]

            sel_time = tf.gather(time_value, ix)
            sel_mat = tf.cast(tf.expand_dims(sel_time, 1) <= tf.expand_dims(time_value, 0), dtype=tf.float32)
            p_lik = tf.gather(score, ix) - tf.math.log(tf.reduce_sum(sel_mat * tf.exp(score), axis=-1))
            loss = -tf.reduce_mean(p_lik)
            return loss
            
    class ci_loss(tf.keras.losses.Loss):
        def __init__(self, name='ci'):
            super(Losses.ci_loss, self).__init__(name=name)

        def call(self, y_true, y_pred):
            
            time_value = tf.squeeze(y_true[:, 0])
            event = tf.squeeze(y_true[:, 1])
            event = tf.cast(event, dtype=tf.bool)
            
            # Expand dimensions to compute pairwise differences
            time_1 = tf.expand_dims(time_value, 0)
            time_2 = tf.expand_dims(time_value, 1)
            event_1 = tf.expand_dims(event, 0)

            # Conditions for concordant pairs
            valid_pairs = tf.logical_and(time_1 < time_2, event_1)

            # Get indices where conditions are True
            ix = tf.where(valid_pairs)

            # Gather scores for valid pairs
            s1 = tf.gather_nd(y_pred, ix[:, :1])
            s2 = tf.gather_nd(y_pred, ix[:, 1:])
            # Calculate concordance
            ci = tf.reduce_mean(tf.cast(s1 < s2, dtype=tf.float32))

            return ci

        def __call__(self, y_true, y_pred, sample_weight=None):
            ##sample weights out of order for multiple cancers, need to reweight based on events. Don't use weighting for now.
            losses = self.call(y_true, y_pred)
            if sample_weight is not None:
                return tf.reduce_sum(losses * sample_weight[:, 0]) / tf.reduce_sum(sample_weight)
            else:
                return tf.reduce_mean(losses)