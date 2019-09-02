import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.backend as tkb  # pylint: disable=E0401


class MonteCarloModelCheckpoint(tk.callbacks.Callback):
    def __init__(self, filepath, monitor, verbose, monte_carlo, validation_x, validation_y):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.monte_carlo = monte_carlo
        self.validation_x = validation_x
        self.validation_y = validation_y

        self.save_best_only = True
        self.best = np.Inf
        self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        monitor_results = []
        for _ in range(self.monte_carlo):
            result = self.model.evaluate(self.validation_x, self.validation_y, batch_size=128, verbose=0)
            result_metrics = {name: value for name, value in zip(self.model.metrics_names, result)}
            monitor_results.append(result_metrics[self.monitor])

        current = np.mean(np.array(monitor_results))
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(f'\nEpoch improved from {self.best:0.5f} to {current:0.5f}')

                self.best = current
                self.model.save(self.filepath, overwrite=True)
            else:
                if self.verbose > 0:
                    print(f'\nEpoch did not improve from {self.best:0.5f} ({current:0.5f})')

        else:
            self.model.save(self.filepath, overwrite=True)


class Trainer:
    def __init__(self, train_set, validation_set, monte_carlo=None):
        self.train_x, self.train_y = train_set
        self.validation_x, self.validation_y = validation_set

        self.monte_carlo = monte_carlo

        self.metrics = [
            self.crossentropy,
            self.mean_square_error,
            self.accuracy,
            self.precision,
            self.recall,
        ]

    @classmethod
    def single_class_split(cls, y_true, y_pred):
        value_true = y_true[:, 0]
        index = tf.to_int32(y_true[:, 1])
        indices_gripper_class = tf.stack([tf.range(tf.shape(index)[0]), index], axis=1)
        value_pred = tf.gather_nd(y_pred, indices_gripper_class)
        return value_true, value_pred

    @classmethod
    def mean_square_error(cls, y_true, y_pred):
        value_true, value_pred = cls.single_class_split(y_true, y_pred)
        return tk.metrics.MSE(value_true, value_pred)

    @classmethod
    def crossentropy(cls, y_true, y_pred):
        value_true, value_pred = cls.single_class_split(y_true, y_pred)
        return tk.metrics.binary_crossentropy(value_true, value_pred)

    @classmethod
    def accuracy(cls, y_true, y_pred):
        value_true, value_pred = cls.single_class_split(y_true, y_pred)
        return tkb.mean(tf.to_float(tkb.equal(tkb.round(value_pred), value_true)))

    @classmethod
    def precision(cls, y_true, y_pred):
        value_true, value_pred = cls.single_class_split(y_true, y_pred)
        true_positives = tkb.sum(tkb.round(tkb.clip(value_true * value_pred, 0, 1)))
        predicted_positives = tkb.sum(tkb.round(tkb.clip(value_pred, 0, 1)))
        return true_positives / (predicted_positives + tkb.epsilon())

    @classmethod
    def recall(cls, y_true, y_pred):
        value_true, value_pred = cls.single_class_split(y_true, y_pred)
        true_positives = tkb.sum(tkb.round(tkb.clip(value_true * value_pred, 0, 1)))
        possible_positives = tkb.sum(tkb.round(tkb.clip(value_true, 0, 1)))
        return true_positives / (possible_positives + tkb.epsilon())

    def train(
            self,
            model,
            load_model=False,
            train_model=False,
            path=None,
            loss_name='crossentropy',
            verbose=1,
            learning_duration=1,
            batch_size=128,
            max_epochs=1000,
            use_beta_checkpoint_path=False,
            **params
        ):
        if load_model:
            model.load_weights(str(path))

        # loss_function = {'crossentropy': self.crossentropy, 'mean_square_error': self.mean_square_error}
        # model.compile(
        #     optimizer=tk.optimizers.Adam(lr=params['lr']),
        #     loss=loss_function[loss_name],
        #     metrics=self.metrics,
        # )


        if train_model:
            checkpoint_path = path if not use_beta_checkpoint_path else path.with_suffix('.beta' + path.suffix)

            if self.monte_carlo:
                checkpointer = MonteCarloModelCheckpoint(
                    str(checkpoint_path),
                    monitor=loss_name,
                    verbose=1,
                    monte_carlo=self.monte_carlo,
                    validation_x=self.validation_x,
                    validation_y=self.validation_y
                )
            else:
                checkpointer = tk.callbacks.ModelCheckpoint(
                    str(checkpoint_path),
                    monitor='val_' + loss_name,
                    verbose=1,
                    save_best_only=True
                )

            callbacks = [
                checkpointer,
                tk.callbacks.EarlyStopping(monitor='val_' + loss_name, patience=80 * learning_duration),
                tk.callbacks.ReduceLROnPlateau(factor=0.25, verbose=1, patience=25 * learning_duration),
            ]

            if load_model:
                evaluation = model.evaluate(self.validation_x, self.validation_y)
                print(model.metrics_names, evaluation)
                callbacks[0].best = evaluation[model.metrics_names.index(loss_name)]

            history = model.fit(
                self.train_x,
                self.train_y,
                batch_size=batch_size,
                epochs=max_epochs,
                shuffle=True,
                validation_data=(self.validation_x, self.validation_y),
                callbacks=callbacks,
                verbose=verbose,
                # sample_weight=np.array([1.0 for i in self.train_y[0]])
            )

            model.load_weights(str(checkpoint_path))
            if use_beta_checkpoint_path:
                model.save(str(path))
        else:
            if not load_model:
                model.save(str(path))
            history = []

        best_metrics = self.evaluate(model)
        for name, value in best_metrics.items():
            print(f'{name}: \t{value}')
        return history, best_metrics

    def evaluate(self, model):
        evaluation = model.evaluate(self.validation_x, self.validation_y)
        return {name: value for name, value in zip(model.metrics_names, evaluation)}

    @classmethod
    def get_flops(cls, path):
        tk.backend.clear_session()
        tk.models.load_model(str(path), compile=False)
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()

        # We use the Keras session graph in the call to the profiler.
        flops = tf.profiler.profile(graph=tk.backend.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops

    @classmethod
    def measure_time(cls, path, images):
        tk.backend.clear_session()
        model = tk.models.load_model(str(path), compile=False)
        model.predict(images)
        model.predict(images)
        start = time.time()
        model.predict(images)
        end = time.time()
        print(f'Time [ms]: {((end - start) * 1000):0.4f}')
        # return end - start
