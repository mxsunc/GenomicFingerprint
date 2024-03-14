import tensorflow as tf
from tensorflow.keras.layers import Lambda
from KerasLayers import Activations, Ragged, Embed, StrandWeight

class InstanceModels:
    class VariantPositionBin:
        def __init__(self, chromosome_embedding_dimension, position_embedding_dimension, default_activation=tf.keras.activations.relu):
            self.chromosome_embedding_dimension = chromosome_embedding_dimension
            self.position_embedding_dimension = position_embedding_dimension
            self.default_activation = default_activation
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            position_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
            position_bin = tf.keras.layers.Input(shape=(), dtype=tf.int32)
            chromosome_input = tf.keras.layers.Input(shape=(), dtype=tf.int32)
            chromosome_emb = Embed(embedding_dimension=self.chromosome_embedding_dimension, trainable=False)
            position_emb = Embed(embedding_dimension=self.position_embedding_dimension, trainable=False, triangular=False)
            pos_loc = tf.keras.layers.Dense(units=64, activation=Activations.ASU())(position_input)
            pos_loc = tf.keras.layers.Dense(units=32, activation=Activations.ARU())(pos_loc)
            pos_loc = tf.concat([position_emb(position_bin), pos_loc], axis=-1)
            pos_loc = tf.keras.layers.Dense(units=96, activation=Activations.ARU())(pos_loc)
            fused = tf.concat([chromosome_emb(chromosome_input), pos_loc], axis=-1)
            latent = tf.keras.layers.Dense(units=128, activation=Activations.ARU())(fused)
            #latent = tf.keras.layers.Dense(units=128, activation='sigmoid')(latent) # 
            self.model = tf.keras.Model(inputs=[position_input, position_bin, chromosome_input], outputs=[latent])

    class VariantSequence:
        def __init__(self, sequence_length, sequence_embedding_dimension, n_strands, convolution_params, fusion_dimension=64, default_activation=tf.keras.activations.relu, use_frame=False, regularization=.01):
            self.sequence_length = sequence_length
            self.sequence_embedding_dimension = sequence_embedding_dimension
            self.convolution_params = convolution_params
            self.default_activation = default_activation
            self.n_strands = n_strands
            self.use_frame = use_frame
            self.fusion_dimension = fusion_dimension
            self.regularization=regularization
            self.model = None
            self.build()

        def build(self, *args, **kwargs):
            five_p = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            three_p = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            ref = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            alt = tf.keras.layers.Input(shape=(self.sequence_length, self.n_strands), dtype=tf.int32)
            strand = tf.keras.layers.Input(shape=(self.n_strands,), dtype=tf.float32)

            # layers of convolution for sequence feature extraction based on conv_params
            features = [[]] * 4
            convolutions = [[]] * 4
            nucleotide_emb = Embed(embedding_dimension=4, trainable=False)
            for index, feature in enumerate([five_p, three_p, ref, alt]):
                convolutions[index] = tf.keras.layers.Conv2D(filters=self.convolution_params[index], kernel_size=[1, self.sequence_length], activation=Activations.ARU())
                
                # apply conv to forward and reverse
                features[index] = tf.stack([convolutions[index](nucleotide_emb(feature)[:, tf.newaxis, :, i, :]) for i in range(self.n_strands)], axis=3)
                # pool over any remaining positions
                features[index] = tf.reduce_max(features[index], axis=[1, 2])

            fused = tf.concat(features, axis=2)
            fused = tf.keras.layers.Dense(units=self.fusion_dimension, activation=self.default_activation)(fused) # 
            fused = tf.reduce_max(StrandWeight(trainable=True, n_features=fused.shape[2])(strand) * fused, axis=1)
            if self.use_frame:
                cds = tf.keras.layers.Input(shape=(3,), dtype=tf.float32)
                frame = tf.concat([strand, cds], axis=-1)
                frame = tf.keras.layers.Dense(units=6, activation=self.default_activation)(frame)
                fused = tf.concat([fused, frame], axis=-1)
                self.model = tf.keras.Model(inputs=[five_p, three_p, ref, alt, strand, cds], outputs=[fused])
            else:
                self.model = tf.keras.Model(inputs=[five_p, three_p, ref, alt, strand], outputs=[fused])

    class PassThrough:
        def __init__(self, shape=None):
            self.shape = shape
            self.model = None
            self.build()

        def build(self, *args, **kwarg):
            input = tf.keras.layers.Input(shape=self.shape, dtype=tf.float32)
            tf.print(input)
            self.model = tf.keras.Model(inputs=[input], outputs=[input])
            

class RaggedModels:
    class AddRandomNoise(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(RaggedModels.AddRandomNoise, self).__init__(**kwargs)

        def call(self, inputs):
            # Add random noise from a uniform distribution over [0, 1)
            noise = tf.random.uniform(tf.shape(inputs), 0, 0.01)
            return inputs + noise

    class MIL:
        def __init__(self, instance_encoders=[], sample_encoders=[], instance_layers=[], sample_layers=[], pooled_layers=[], output_dim=1, output_type='classification', mode='attention', pooling='sum', regularization=.0, fusion='after', mil_hidden=[32, 16]):
            self.instance_encoders, self.sample_encoders, self.instance_layers, self.sample_layers, self.pooled_layers, self.output_dim, self.output_type, self.mode, self.pooling, self.regularization, self.fusion, self.mil_hidden = instance_encoders, sample_encoders, instance_layers, sample_layers, pooled_layers, output_dim, output_type, mode, pooling, regularization, fusion, mil_hidden
            self.model, self.attention_model = None, None
            self.build()

        def build(self):
            ragged_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape, dtype=input_tensor.dtype, ragged=True) for input_tensor in encoder.inputs] for encoder in self.instance_encoders]
            sample_inputs = [[tf.keras.layers.Input(shape=input_tensor.shape[1:], dtype=input_tensor.dtype) for input_tensor in encoder.inputs] for encoder in self.sample_encoders]

            ##sample level model encodings
            if self.sample_encoders != []:
                sample_encodings = [encoder(sample_input) for sample_input, encoder in zip(sample_inputs, self.sample_encoders)]
                sample_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(sample_encodings)

            if self.instance_encoders != []:
                
                ragged_encodings = [Ragged.MapFlatValues(encoder)(ragged_input) for ragged_input, encoder in zip(ragged_inputs, self.instance_encoders)]
                # flatten encoders if needed
                
                ragged_encodings = [Ragged.MapFlatValues(tf.keras.layers.Flatten())(ragged_encoding) for ragged_encoding in ragged_encodings]
                # based on the design of the input and graph instances can be fused prior to bag aggregation
                ragged_fused = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=2))(ragged_encodings)
                
                if self.sample_encoders != []:
                    if self.fusion == 'before':
                        ragged_hidden = [Ragged.Dense(units=64, activation=tf.keras.activations.relu)((ragged_fused, sample_fused))]
                    else:
                        ragged_hidden = [ragged_fused]
                else:
                    ragged_hidden = [ragged_fused]

                for i in self.instance_layers:
                    ragged_hidden.append(Ragged.MapFlatValues(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu))(ragged_hidden[-1]))

                if self.mode == 'attention':
                    if self.pooling == 'both':
                        pooling, ragged_attention_weights = Ragged.Attention(pooling='mean', regularization=self.regularization)(ragged_hidden[-1])
                        pooled_hidden = [tf.concat([pooling[:, 0, :], tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(ragged_attention_weights)], axis=-1)]
                    elif self.pooling == 'dynamic':
                        pooling_1, ragged_attention_weights_1 = Ragged.Attention(pooling='mean', regularization=self.regularization)(ragged_hidden[-1])
                        instance_ragged_fused = Ragged.Dense(units=32, activation=tf.keras.activations.relu)((ragged_hidden[-1], pooling_1[:, 0, :]))
                        pooling_2, ragged_attention_weights = Ragged.Attention(pooling='dynamic', regularization=self.regularization)([ragged_hidden[-1], instance_ragged_fused])
                        pooled_hidden = [pooling_2[:, 0, :]]
                    else:
                        pooling, ragged_attention_weights = Ragged.Attention(pooling=self.pooling, regularization=self.regularization)(ragged_hidden[-1])
                        pooled_hidden = [pooling[:, 0, :]]
                else:
                    if self.pooling == 'mean':
                        pooling = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=ragged_hidden[-1].ragged_rank))(ragged_hidden[-1])
                    else:
                        pooling = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=ragged_hidden[-1].ragged_rank))(ragged_hidden[-1])
                    pooled_hidden = [pooling]

                for i in self.pooled_layers:
                    pooled_hidden.append(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu)(pooled_hidden[-1]))

            if self.sample_encoders != []:
                if self.fusion == 'after':
                    if self.instance_encoders != []:
                        fused = [tf.concat([pooled_hidden[-1], sample_fused], axis=-1)]
                    else:
                        fused = [sample_fused]

                else:
                    fused = [pooled_hidden[-1]]

            else:
                fused = [pooled_hidden[-1]]

            for i in self.sample_layers:
                fused.append(tf.keras.layers.Dense(units=i, activation=tf.keras.activations.relu)(fused[-1]))

            for i in self.mil_hidden:
                fused.append(tf.keras.layers.Dense(units=i, activation='relu')(fused[-1]))
            fused = fused[-1]

            if self.output_type == 'quantiles':
                output_layers = (8, 1)
                point_estimate, lower_bound, upper_bound = list(), list(), list()
                for i in range(len(output_layers)):
                    point_estimate.append(tf.keras.layers.Dense(units=output_layers[i], activation=None if i == (len(output_layers) - 1) else tf.keras.activations.softplus)(fused if i == 0 else point_estimate[-1]))

                for l in [lower_bound, upper_bound]:
                    for i in range(len(output_layers)):
                        l.append(tf.keras.layers.Dense(units=output_layers[i], activation=tf.keras.activations.softplus)(fused if i == 0 else l[-1]))

                output_tensor = tf.keras.activations.softplus(tf.concat([point_estimate[-1] - lower_bound[-1], point_estimate[-1], point_estimate[-1] + upper_bound[-1]], axis=1))

            elif self.output_type == 'survival':
                output_layers = (8, 4, 1)  # Define the size of each layer
                pred = list()  # Initialize list to store layer outputs
                for i in range(len(output_layers)):
                    # Apply sigmoid activation function to the last layer
                    if i == (len(output_layers) - 1):
                        activation_function = tf.keras.activations.sigmoid
                    else:
                        activation_function = tf.keras.activations.relu
                    # Create Dense layer with appropriate activation function
                    layer = tf.keras.layers.Dense(units=output_layers[i], activation=activation_function)(fused if i == 0 else pred[-1])
                    pred.append(layer)  # Add layer output to the list
                output_tensor = (pred[-1])  # Final layer output as the model's output tensor

            elif self.output_type == 'regression':
                ##assumes log transformed output
                pred = tf.keras.layers.Dense(units=self.output_dim, activation='sigmoid')(fused)
                output_tensor = pred
                #tf.math.log(pred + 1)

            elif self.output_type == 'anlulogits':
                output_tensor = tf.keras.layers.Dense(units=self.output_dim, activation=Activations.ARU())(fused)

            elif self.output_type == 'classification_probability':
                probabilities = tf.keras.layers.Dense(units=self.output_dim, activation=Activations.ARU())(fused)
                probabilities = probabilities / tf.expand_dims(tf.reduce_sum(probabilities, axis=-1), axis=-1)
                output_tensor = probabilities

            else:
                output_tensor = tf.keras.layers.Dense(units=self.output_dim, activation='sigmoid')(fused)
            self.model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[output_tensor])
            if self.mode == 'attention':
                self.attention_model = tf.keras.Model(inputs=ragged_inputs + sample_inputs, outputs=[ragged_attention_weights])
