import tensorflow.keras as keras
import tensorflow as tf
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model

class SATModel(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(SATModel, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.k = 1.0

    def compile(self, gen_optimizer, dis_optimizer, gen_loss, dis_loss, **kwargs):
        super(SATModel, self).compile(**kwargs)
        self.optimizer_generator = gen_optimizer
        self.optimizer_discriminator = dis_optimizer
        self.loss_generator = gen_loss
        self.loss_discriminator = dis_loss
        #self.loss_sum = keras.metrics.Sum()

    # train step for conditional target adversarial training
    @tf.function
    def train_step(self, data):

        x_s = data[0]
        y_s = data[1]

        loss_names_gen = []
        loss_gen = []
        loss_sum_gen = 0

        # run prediction to sample latent space
        with tf.GradientTape() as tape:
            predicts_gen = self.generator(x_s)
            for ldx, loss_func in enumerate(self.loss_generator):
                loss_names_gen.append(loss_func)
                print(self.loss_generator[loss_func])
                y_now = tf.convert_to_tensor(y_s[ldx], dtype=tf.float32)
                loss = self.loss_generator[loss_func](y_now, predicts_gen[ldx])
                #loss = self.loss_generator[loss_func](y_now, filtered_predictions[ldx])
                print(loss)
                loss_gen.append(loss)
                loss_sum_gen += loss

            print('rec')

            loss_names_gen.append('rec')
            print(self.loss_discriminator)
            y_s_0 = tf.convert_to_tensor(y_s[0], dtype=tf.float32)
            gt_targets, anchor_validity = tf.split(y_s_0, num_or_size_splits=[16, 1], axis=2)
            generated = tf.concat([predicts_gen, anchor_validity], axis=2)
            loss = self.loss_discriminator(generated, predicts_gen[3])
            # loss = self.loss_generator[loss_func](y_now, filtered_predictions[ldx])
            print(loss)
            loss_gen.append(loss)
            loss_sum_gen += loss

        # Sample random points in the latent space
        batch_size = tf.shape(x_s)[0]

        points = predicts_gen[0]
        #locations = predicts_gen[1]
        #masks = predicts_gen[2]
        recon = predicts_gen[3]
        featuresP3 = predicts_gen[4]
        featuresP4 = predicts_gen[5]
        featuresP5 = predicts_gen[6]

        #cls_shape = tf.shape(locations)[2]
        # split predictions to pyramid levels
        pointsP3, pointsP4, pointsP5 = tf.split(points, num_or_size_splits=[43200, 10800, 2700], axis=1)
        pointsP3_re = tf.reshape(pointsP3, (batch_size, 60, 80, 9 * 16))
        pointsP4_re = tf.reshape(pointsP4, (batch_size, 30, 40, 9 * 16))
        pointsP5_re = tf.reshape(pointsP5, (batch_size, 15, 20, 9 * 16))

        # split predictions to pyramid levels
        gtP3, gtP4, gtP5 = tf.split(y_s[0], num_or_size_splits=[43200, 10800, 2700], axis=1)
        gtP3_re = tf.reshape(gtP3, (batch_size, 60, 80, 9 * 16))
        gtP4_re = tf.reshape(gtP4, (batch_size, 30, 40, 9 * 16))
        gtP5_re = tf.reshape(gtP5, (batch_size, 15, 20, 9 * 16))

        featuresP3_re = tf.reshape(featuresP3, (batch_size, 60, 80, 256))
        featuresP4_re = tf.reshape(featuresP4, (batch_size, 30, 40, 256))
        featuresP5_re = tf.reshape(featuresP5, (batch_size, 15, 20, 256))

        patchP3_gen = tf.concat([featuresP3_re, pointsP3_re], axis=3)
        patchP4_gen = tf.concat([featuresP4_re, pointsP4_re], axis=3)
        patchP5_gen = tf.concat([featuresP5_re, pointsP5_re], axis=3)

        patchP3_gt = tf.concat([featuresP3_re, gtP3_re], axis=3)
        patchP4_gt = tf.concat([featuresP4_re, gtP4_re], axis=3)
        patchP5_gt = tf.concat([featuresP5_re, gtP5_re], axis=3)

        disc_patch_gen = [patchP3_gen, patchP4_gen, patchP5_gen]
        disc_patch_gt = [patchP3_gt, patchP4_gt, patchP5_gt]

        # Forward 4: Discriminator reconstructs gt
        disc_patch_real = []
        disc_patch_fake = []
        #loss_dis = 0

        with tf.GradientTape() as tape:
            for idx, patch in enumerate(disc_patch_gt):
                rec_level = self.discriminator(patch)
                rec_level = tf.reshape(rec_level, (batch_size, tf.shape(rec_level)[1] * tf.shape(rec_level)[2], 16))
                disc_patch_real.append(rec_level)

            recon_real = tf.concat(disc_patch_real, axis=2)
            loss_real = self.loss_discriminator(y_s[0], recon_real)

            for idx, patch in enumerate(disc_patch_gen):
                rec_level = self.discriminator(patch)
                rec_level = tf.reshape(rec_level, (batch_size, tf.shape(rec_level)[1] * tf.shape(rec_level)[2], 16))
                disc_patch_fake.append(rec_level)

            recon_fake = tf.concat(disc_patch_real, axis=2)
            loss_fake = self.loss_discriminator(points, recon_fake)

            loss_dis = loss_real - self.k * loss_fake

            #self.k = self.k + self.lambda * (self.gamma * loss_real - loss_fake)

        # Accumulate discriminator
        grads_dis = tape.gradient(loss_dis, self.discriminator.trainable_weights)
        # Update discriminator
        self.optimizer_discriminator.apply_gradients(zip(grads_dis, self.discriminator.trainable_weights))

        # Accumulate generator
        grads_gen = tape.gradient(loss_sum_gen, self.generator.trainable_weights)
        # Update generator
        self.optimizer_generator.apply_gradients(zip(grads_gen, self.pyrapose.trainable_weights))

        return_losses = {}
        return_losses["loss"] = loss_sum_gen + loss_dis
        for name, loss in zip(loss_names_gen, loss_gen):
            return_losses[name] = loss
        return_losses['disc'] = loss_dis

        return return_losses

    def call(self, inputs, training=False):
        x = self.generator(inputs[0])
        if training:
            x = self.generator(inputs[0])
        return x



