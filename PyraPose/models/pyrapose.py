import keras
import tensorflow as tf
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


class CustomModel(tf.keras.Model):
    def __init__(self, pyrapose, discriminator):
        super(CustomModel, self).__init__()
        self.discriminator = discriminator
        self.pyrapose = pyrapose

    def compile(self, omni_optimizer, gen_loss, dis_loss, **kwargs):
        super(CustomModel, self).compile(**kwargs)
        self.omni_optimizer = omni_optimizer
        self.loss_generator = gen_loss
        self.loss_discriminator = dis_loss

    def train_step(self, data):
        #if isinstance(data, tuple):
        x_s = data[0]
        y_s = data[1]
        x_t = data[2]
        # Sample random points in the latent space
        batch_size = tf.shape(x_s)[0]
        valid = tf.ones((batch_size, 60, 80,  2))
        fake1 = tf.zeros((batch_size, 60, 80, 1))
        fake2 = tf.ones((batch_size, 60, 80, 1 ))
        fake = tf.concat([fake1, fake2], axis=3)
        labels = tf.concat([valid, fake], axis=0)

        # from Chollet
        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))

        x_st = tf.concat([x_s, x_t], axis=0)
        with tf.GradientTape() as tape_gen:
            predicts_gen = self.pyrapose.predict(x_st, batch_size=None, steps=1)

        points = predicts_gen[0]
        locations = predicts_gen[1]
        masks = predicts_gen[2]
        domain = predicts_gen[3]
        features = predicts_gen[4]

        source_points, target_points = tf.split(points, num_or_size_splits=2, axis=0)
        source_locations, target_locations = tf.split(locations, num_or_size_splits=2, axis=0)
        source_mask, target_mask = tf.split(masks, num_or_size_splits=2, axis=0)
        source_domain, target_domain = tf.split(domain, num_or_size_splits=2, axis=0)
        source_features, target_features = tf.split(features, num_or_size_splits=2, axis=0)

        cls_shape = tf.shape(source_mask)[2]
        source_mask_re = tf.reshape(source_mask, (batch_size, 60, 80, cls_shape))
        target_mask_re = tf.reshape(target_mask, (batch_size, 60, 80, cls_shape))

        source_patch = tf.concat([source_features, source_mask_re], axis=3)
        target_patch = tf.concat([target_features, target_mask_re], axis=3)
        disc_patch = tf.concat([target_patch, source_patch], axis=0)

        filtered_predictions = [source_points, source_locations, source_mask, source_domain]
        losses = []
        with tf.GradientTape() as tape_dis:
            domain = self.discriminator(disc_patch)
            d_loss = self.loss_discriminator(labels, domain)
        with tape_gen:
            for ldx, task in enumerate(y_s):
                losses.append(self.loss_generator[ldx](task, filtered_predictions[ldx]))
            #g_loss = self.loss_generator(y_s, )

        grads_dis = tape_dis.gradient(d_loss, self.discriminator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_dis, self.discriminator.trainable_weights))

        grads_gen = tape_gen.gradient(losses, self.pyrapose.trainable_weights)
        for grads_loss in enumerate(grads_gen):
            self.optimizer.apply_gradients(zip(grads_loss, self.pyrapose.trainable_weights))

        #'3Dbox', 'cls', 'mask', 'domain', 'P3'
        return {"d_loss": d_loss, "g_loss": g_loss}

    def call(self, inputs):
        loss = self.train_step(inputs)
        return loss


