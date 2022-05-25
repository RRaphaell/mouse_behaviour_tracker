import os
import tensorflow as tf



class myCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_iou_map', save_dir = ".", this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        self.save_dir = save_dir
        
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()
                self.model.save(os.path.join(self.save_dir,"best_model.h5"))

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights= self.model.get_weights()
                self.model.save(os.path.join(self.save_dir,"best_model.h5"))
        
        if(logs.get('val_iou_map') > 0.98):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True