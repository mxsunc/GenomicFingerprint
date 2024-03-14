import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

def rescale_batch_weights(X, y, w):
    return X, y, (w / tf.reduce_sum(w)) * tf.cast(tf.shape(w)[0], tf.float64)

class Map:
    class LoadBatchByIndices:
        def loader(self):
            raise NotImplementedError

        def __call__(self, sample_idx, ragged_output=True):
            # flat_values and additional_args together should be the input into the ragged_constructor of the loader
            flat_values, *additional_args = tf.py_function(self.loader, [sample_idx], self.tf_output_types)
            flat_values.set_shape((None,) + self.inner_shape)

            if ragged_output:
                return self.ragged_constructor(flat_values, *additional_args)
            else:
                return flat_values

    class FromNumpy(LoadBatchByIndices):
        def __init__(self, data, data_type):
            self.data = data
            self.tf_output_types = [data_type, tf.int32]
            self.inner_shape = data[0].shape[1:]
            self.ragged_constructor = tf.RaggedTensor.from_row_lengths

        def loader(self, idx):
            batch = list()
            for i in idx.numpy():
                batch.append(self.data[i])
            return np.concatenate(batch, axis=0), np.array([v.shape[0] for v in batch])
        
class DelayedEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, start_epoch=200, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch  # Epoch to start monitoring
        self.wait_since_start = 0  # Counter to track waiting since start_epoch

    def on_epoch_end(self, epoch, logs=None):
        # Override on_epoch_end to delay early stopping
        if epoch < self.start_epoch:
            # Before start_epoch, reset wait counter and best weights
            self.wait = 0
            self.stopped_epoch = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            # After reaching start_epoch, proceed with normal early stopping checks
            if self.wait_since_start == 0:  # First epoch to start monitoring
                print(f"Starting to monitor for early stopping at epoch {epoch+1}")
            self.wait_since_start += 1  # Increment counter
            super().on_epoch_end(epoch, logs)