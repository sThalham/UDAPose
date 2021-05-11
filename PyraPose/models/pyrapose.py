import tensorflow.keras as keras
import tensorflow as tf
import sys
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
    #@tf.function
    def train_step(self, data):

        # x_s = data[0]['x']
        # y_s = data[0]['y']
        # x_t = data[0]['domain']

        x_s = data[0]
        y_s = data[1]

        loss_names = []
        losses = []
        loss_sum = 0
        d_loss_sum = 0

        # Sample random points in the latent space
        batch_size = tf.shape(x_s)[0]
        #tf.print(tf.shape(x_s))
        # UDA mask

        # from Chollet
        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))
        # x_st = tf.concat([x_s, x_t], axis=0)

        with tf.GradientTape() as tape:
            predicts_gen = self.generator(x_s)

            for ldx, loss_func in enumerate(self.loss_generator):
                # print(loss_func)
                loss_names.append(loss_func)
                y_now = tf.convert_to_tensor(y_s[ldx], dtype=tf.float32)
                loss = self.loss_generator[loss_func](y_now, predicts_gen[ldx])

                losses.append(loss)
                loss_sum += loss

        # compute gradients of pyrapose with discriminator frozen
        grads_gen = tape.gradient(loss_sum, self.generator.trainable_weights)
        self.optimizer_generator.apply_gradients(zip(grads_gen, self.generator.trainable_weights))

        gt = y_s[0]
        generated = predicts_gen[0]
        generated_gt = tf.concat([generated, gt[:, :, -1]], axis=2)

        locations = predicts_gen[1]
        masks = predicts_gen[2]
        reconstruction = predicts_gen[3]
        featuresP3 = predicts_gen[4]
        featuresP4 = predicts_gen[5]
        featuresP5 = predicts_gen[6]

        # inputs discriminator for feature map conditioned on predicted pose
        genP3, genP4, genP5 = tf.split(generated, num_or_size_splits=[43200, 10800, 2700], axis=1)
        genP3_re = tf.reshape(genP3, (batch_size, 60, 80, 9 * 16))
        genP4_re = tf.reshape(genP4, (batch_size, 30, 40, 9 * 16))
        genP5_re = tf.reshape(genP5, (batch_size, 15, 20, 9 * 16))

        gen_patchP3 = tf.concat([featuresP3, genP3_re], axis=3)
        gen_patchP4 = tf.concat([featuresP4, genP4_re], axis=3)
        gen_patchP5 = tf.concat([featuresP5, genP5_re], axis=3)

        gtP3, gtP4, gtP5 = tf.split(gt[:, :, :-1], num_or_size_splits=[43200, 10800, 2700], axis=1)
        gtP3_re = tf.reshape(gtP3, (batch_size, 60, 80, 9 * 16))
        gtP4_re = tf.reshape(gtP4, (batch_size, 30, 40, 9 * 16))
        gtP5_re = tf.reshape(gtP5, (batch_size, 15, 20, 9 * 16))

        gt_patchP3 = tf.concat([featuresP3, gtP3_re], axis=3)
        gt_patchP4 = tf.concat([featuresP4, gtP4_re], axis=3)
        gt_patchP5 = tf.concat([featuresP5, gtP5_re], axis=3)

        # input patches
        disc_patches_gt = [gt_patchP3, gt_patchP4, gt_patchP5]
        disc_patches_gen = [gen_patchP3, gen_patchP4, gen_patchP5]
        disc_reso = [[60, 80], [30, 40], [15, 20]]

        # target patches
        #real = tf.ones((batch_size, 56700, 1))
        #fake = tf.zeros((batch_size, 56700, 1))
        #points_anno = y_s[0][:, :, -1]
        #points_anno = tf.expand_dims(points_anno, axis=2)
        #real_targets_dis = tf.concat([gt, points_anno], axis=2)
        #fake_targets_dis = tf.concat([generated, points_anno], axis=2)

        # run and train discriminator
        recon_gt = []
        recon_gen = []
        with tf.GradientTape() as tape:
        #with tape:
            for ddx, disc_map in enumerate(disc_patches_gt):
                gt_rec_lv = self.discriminator(disc_map)
                gen_rec_lv = self.discriminator(disc_patches_gen[ddx])
                # domain = tf.reshape(domain, (batch_size * 2, keras.backend.int_shape(x_s)[1] * keras.backend.int_shape(x_s)[2], 1))
                gt_rec_lv = tf.reshape(gt_rec_lv, (batch_size * 2, disc_reso[ddx][0] * disc_reso[ddx][1] * 9, 1))
                gen_rec_lv = tf.reshape(gen_rec_lv, (batch_size * 2, disc_reso[ddx][0] * disc_reso[ddx][1] * 9, 1))
                recon_gt.append(gt_rec_lv)
                recon_gen.append(gen_rec_lv)
            predictions_gt = tf.concat([recon_gt[0], recon_gt[1], recon_gt[2]], axis=1)
            loss_real = self.loss_discriminator(gt, predictions_gt)
            predictions_gen = tf.concat([recon_gen[0], recon_gen[1], recon_gen[2]], axis=1)
            loss_fake = self.loss_discriminator(generated_gt, predictions_gen)
            loss_d = loss_real - loss_fake

            # TODO hyperparameters for loss balancing

        grads_dis = tape.gradient(loss_d, self.discriminator.trainable_weights)

        self.optimizer_discriminator.apply_gradients(zip(grads_dis, self.discriminator.trainable_weights))
        #self.optimizer_generator.apply_gradients(zip(grads_gen, self.generator.trainable_weights))

        return_losses = {}
        return_losses["loss"] = loss_sum
        for name, loss in zip(loss_names, losses):
            return_losses[name] = loss

        return return_losses

    def call(self, inputs, training=False):
        x = self.pyrapose(inputs[0])
        if training:
            x = self.pyrapose(inputs[0])
        return x




