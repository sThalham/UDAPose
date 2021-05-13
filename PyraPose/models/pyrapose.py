import tensorflow.keras as keras
import tensorflow as tf
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


class CustomModel(tf.keras.Model):
    def __init__(self, pyrapose, disc_model, discriminator):
        super(CustomModel, self).__init__()
        self.discriminator = discriminator
        self.pyrapose = pyrapose
        self.model_disc = disc_model

    def compile(self, gen_optimizer, dis_optimizer, gen_loss, dis_loss, **kwargs):
        self.model_disc.get_layer('disc').trainable = False
        self.model_disc.get_layer('disc_').trainable = False
        super(CustomModel, self).compile(**kwargs)
        self.optimizer_generator = gen_optimizer
        self.optimizer_discriminator = dis_optimizer
        self.loss_generator = gen_loss
        self.loss_discriminator = dis_loss
        #self.pyrapose.get_layer('disc').trainable = False
        #self.pyrapose.get_layer('disc_').trainable = False
        #for layer in self.pyrapose.get_layer('disc_').layers:
        #    print(layer.name, layer.trainable)

    #@tf.function
    def train_step(self, data):

        #x_s = data[0]['x']
        #y_s = data[0]['y']
        #x_t = data[0]['domain']

        x_s = data[0]
        y_s = data[1]
        x_t = data[2]

        loss_names = []
        losses = []
        loss_sum = 0
        d_loss_sum = 0

        # Sample random points in the latent space
        batch_size = tf.shape(x_s)[0]
        # UDA mask

        # from Chollet
        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))
        #x_st = tf.concat([x_s, x_t], axis=0)

        #print('pyrapose.py::discriminator')
        #for layer in self.discriminator.layers:
        #    print(layer.name, layer.trainable)

        with tf.GradientTape(persistent=True) as tape:
            predicts_gen = self.pyrapose(x_s)
            for ldx, loss_func in enumerate(self.loss_generator):
                print(loss_func)
                loss_names.append(loss_func)
                y_now = tf.convert_to_tensor(y_s[ldx], dtype=tf.float32)
                #print(keras.backend.int_shape(y_s[ldx]))
                #print(keras.backend.int_shape(predicts_gen[ldx]))
                loss = self.loss_generator[loss_func](y_now, predicts_gen[ldx])
                losses.append(loss)
                loss_sum += loss

            predicts_dis = self.model_disc(x_s)
            loss_names.append('domain')
            y_now = tf.convert_to_tensor(y_s[3], dtype=tf.float32)
            # print(keras.backend.int_shape(y_s[ldx]))
            # print(keras.backend.int_shape(predicts_gen[ldx]))
            loss = self.loss_discriminator(y_now, predicts_dis)
            losses.append(loss)
            loss_sum += loss

        # compute gradients of pyrapose with discriminator frozen
        grads_gen = tape.gradient(loss_sum, self.pyrapose.trainable_weights)
        #self.optimizer_generator.apply_gradients(zip(grads_gen, self.pyrapose.trainable_weights))

        predicts_target = self.pyrapose(x_t)
        target_points = predicts_target[0]
        target_locations = predicts_target[1]
        masks = predicts_gen[2]
        #domain = predicts_gen[3]
        target_featuresP3 = predicts_target[3]
        target_featuresP4 = predicts_target[4]
        target_featuresP5 = predicts_target[5]

        source_points = predicts_gen[0]
        locations = predicts_gen[1]
        masks = predicts_gen[2]
        #domain = predicts_gen[3]
        source_featuresP3 = predicts_gen[3]
        source_featuresP4 = predicts_gen[4]
        source_featuresP5 = predicts_gen[5]

        # discriminator for feature map conditioned on predicted pose
        source_pointsP3, source_pointsP4, source_pointsP5 = tf.split(source_points, num_or_size_splits=[43200, 10800, 2700], axis=1)
        source_pointsP3_re = tf.reshape(source_pointsP3, (batch_size, 60, 80, 9 * 16))
        source_pointsP4_re = tf.reshape(source_pointsP4, (batch_size, 30, 40, 9 * 16))
        source_pointsP5_re = tf.reshape(source_pointsP5, (batch_size, 15, 20, 9 * 16))

        target_pointsP3, target_pointsP4, target_pointsP5 = tf.split(target_points, num_or_size_splits=[43200, 10800, 2700], axis=1)
        target_pointsP3_re = tf.reshape(target_pointsP3, (batch_size, 60, 80, 9 * 16))
        target_pointsP4_re = tf.reshape(target_pointsP4, (batch_size, 30, 40, 9 * 16))
        target_pointsP5_re = tf.reshape(target_pointsP5, (batch_size, 15, 20, 9 * 16))

        source_patchP3 = tf.concat([source_featuresP3, source_pointsP3_re], axis=3)
        target_patchP3 = tf.concat([target_featuresP3, target_pointsP3_re], axis=3)
        disc_patchP3 = tf.concat([target_patchP3, source_patchP3], axis=0)

        source_patchP4 = tf.concat([source_featuresP4, source_pointsP4_re], axis=3)
        target_patchP4 = tf.concat([target_featuresP4, target_pointsP4_re], axis=3)
        disc_patchP4 = tf.concat([target_patchP4, source_patchP4], axis=0)

        source_patchP5 = tf.concat([source_featuresP5, source_pointsP5_re], axis=3)
        target_patchP5 = tf.concat([target_featuresP5, target_pointsP5_re], axis=3)
        disc_patchP5 = tf.concat([target_patchP5, source_patchP5], axis=0)

        disc_patch_fake = [source_patchP3, source_patchP4, source_patchP5]
        disc_patch_real = [target_patchP3, target_patchP4, target_patchP5]
        #disc_patches = [disc_patchP3, disc_patchP4, disc_patchP5]
        disc_reso = [[60, 80], [30, 40], [15, 20]]

        # UDA pose
        real = tf.ones((batch_size, 56700,  1))
        fake = tf.zeros((batch_size, 56700, 1))

        target_locations_red = tf.math.reduce_max(target_locations, axis=2, keepdims=False, name=None) # find max value along cls dimension
        sst_temp = tf.math.reduce_max(tf.math.reduce_max(target_locations_red, axis=1, keepdims=False, name=None), axis=0, keepdims=False, name=None) * 0.8
        lr_scale = tf.cast(tf.cast(sst_temp, dtype=tf.int32), dtype=tf.float32) # gatekeeping at its finest
        tf.print(' ')
        tf.print(sst_temp, lr_scale)
        #tf.print(tf.math.reduce_sum(target_locations_red))
        target_locations_red = tf.expand_dims(target_locations_red, axis=2)
        #print(target_locations_red)
        real_pseudo_anno_cls = tf.math.greater(target_locations_red, sst_temp*0.8)
        #tf.print(real_pseudo_anno_cls)
        real_anno = tf.ones((batch_size, 56700, 1))
        #tf.print(tf.math.reduce_sum(real_anno))
        real_anno = real_anno * tf.cast(real_pseudo_anno_cls, dtype=tf.float32)
        #tf.print(tf.math.reduce_sum(real_anno))
        #real_targets_dis = tf.concat([real, real_anno], axis=2)

        fake_anno = y_s[0][:, :, -1]
        fake_anno = tf.expand_dims(fake_anno, axis=2)
        fake_targets_dis = tf.concat([fake, fake_anno], axis=2)
        real_targets_dis = tf.concat([real, fake_anno], axis=2)

        #disc_labels = tf.concat([real_targets_dis, fake_targets_dis], axis=0)
        #print('disc_labels: ', disc_labels)

        dis_real = []
        dis_fake = []
        #with tf.GradientTape() as tape:
        with tape:
            for ddx, disc_map in enumerate(disc_patch_real):
                domain = self.discriminator(disc_map)
                domain = tf.reshape(domain, (batch_size, disc_reso[ddx][0] * disc_reso[ddx][1] * 9, 1))
                dis_real.append(domain)
            disc_real = tf.concat([dis_real[0], dis_real[1], dis_real[2]], axis=1)
            loss_real = self.loss_discriminator(real_targets_dis, disc_real)

            for ddx, disc_map in enumerate(disc_patch_fake):
                domain = self.discriminator(disc_map)
                # domain = tf.reshape(domain, (batch_size * 2, keras.backend.int_shape(x_s)[1] * keras.backend.int_shape(x_s)[2], 1))
                domain = tf.reshape(domain, (batch_size, disc_reso[ddx][0] * disc_reso[ddx][1] * 9, 1))
                dis_fake.append(domain)
            disc_fake = tf.concat([dis_fake[0], dis_fake[1], dis_fake[2]], axis=1)
            loss_fake = self.loss_discriminator(fake_targets_dis, disc_fake)
            loss_d = loss_real + loss_fake

        #tf.print(loss_d)

        grads_dis = tape.gradient(loss_d, self.discriminator.trainable_weights)

        self.optimizer_discriminator.apply_gradients(zip(grads_dis, self.discriminator.trainable_weights))

        self.optimizer_generator.apply_gradients(zip(grads_gen, self.pyrapose.trainable_weights))
        self.optimizer_discriminator.apply_gradients(zip(grads_gen, self.model_disc.trainable_weights))

        del tape

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



