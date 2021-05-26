import tensorflow.keras as keras
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

        #self.exp_mvg_avg = tf.Variable(0.0)
        #self.lr_scale = tf.Variable(0.0)
        #self.source_scale = tf.Variable(1.0)
        #self.target_scale = tf.Variable(1.0)

    def compile(self, gen_optimizer, dis_optimizer, gen_loss, dis_loss, **kwargs):

        super(CustomModel, self).compile(**kwargs)
        self.optimizer_generator = gen_optimizer
        self.optimizer_discriminator = dis_optimizer
        self.loss_generator = gen_loss
        self.loss_discriminator = dis_loss
        #self.pyrapose.get_layer('disc').trainable = False
        #self.pyrapose.get_layer('disc_').trainable = False
        #for layer in self.discriminator.layers:
        #    print(layer.name, layer.weights)

    def set_lr_scale(self):
        self.lr_scale = 1.0

    def set_exp_mvg_avg(self, val):
        self.exp_mvg_avg = val

    def do_not_set_lr_scale(self):
        self.lr_scale = 0.0

    def return_1(self):
        return 1.0

    def return_0(self):
        return 0.0

    @tf.function
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

        batch_size = tf.shape(x_s)[0]
        disc_reso = [[60, 80], [30, 40], [15, 20]]

        source_domain = tf.concat([tf.zeros((batch_size, 6300, 1)), tf.ones((batch_size, 6300, 1))], axis=2)
        target_domain = tf.ones((batch_size, 6300, 2))

        predicts_target = self.pyrapose(x_t)
        target_points = predicts_target[0]
        target_locations = predicts_target[1]
        # masks = predicts_target[2]
        target_featuresP3 = predicts_target[3]
        target_featuresP4 = predicts_target[4]
        target_featuresP5 = predicts_target[5]

        target_patches = [target_featuresP3, target_featuresP4, target_featuresP5]

        dis_fake = []

        #inputs = tf.concat([x_s, x_t], axis=0)

        with tf.GradientTape(persistent=True) as tape:
            predicts_source = self.pyrapose(x_s)

            for ldx, loss_func in enumerate(self.loss_generator):
                loss_names.append(loss_func)
                y_now = tf.convert_to_tensor(y_s[ldx], dtype=tf.float32)
                loss = self.loss_generator[loss_func](y_now, predicts_source[ldx])
                losses.append(loss)
                loss_sum += loss

            # track gradient
            source_points = predicts_source[0]
            #locations = predicts_source[1]
            #masks = predicts_source[2]
            source_featuresP3 = predicts_source[3]
            source_featuresP4 = predicts_source[4]
            source_featuresP5 = predicts_source[5]

            source_patches = [source_featuresP3, source_featuresP4, source_featuresP5]

            for ddx, disc_map in enumerate(source_patches):
                source_discrimination = self.discriminator(disc_map)
                source_discrimination = tf.reshape(source_discrimination,
                                                   (batch_size, disc_reso[ddx][0] * disc_reso[ddx][1], 1))
                dis_fake.append(source_discrimination)
            disc_fake = tf.concat([dis_fake[0], dis_fake[1], dis_fake[2]], axis=1)

            loss_fake = self.loss_discriminator(target_domain, disc_fake)
            #tf.print(keras.backend.int_shape(loss_fake))
            loss_names.append('domain')
            losses.append(loss_fake)
            #loss_fake += loss_fake

        grads_fake = tape.gradient(loss_fake, self.pyrapose.trainable_weights)
        grads_gen = tape.gradient(loss_sum, self.pyrapose.trainable_weights)

        dis_source = []
        dis_target = []

        with tape:

            for ddx, disc_map in enumerate(target_patches):
                domain = self.discriminator(disc_map)
                domain = tf.reshape(domain, (batch_size, disc_reso[ddx][0] * disc_reso[ddx][1], 1))
                dis_target.append(domain)
            disc_target = tf.concat([dis_target[0], dis_target[1], dis_target[2]], axis=1)
            loss_target = self.loss_discriminator(target_domain, disc_target)

            for ddx, disc_map in enumerate(source_patches):
                domain = self.discriminator(disc_map)
                domain = tf.reshape(domain, (batch_size, disc_reso[ddx][0] * disc_reso[ddx][1], 1))
                dis_source.append(domain)
            disc_source = tf.concat([dis_source[0], dis_source[1], dis_source[2]], axis=1)
            loss_source = self.loss_discriminator(source_domain, disc_source)

            #self.source_scale.assign(loss_target / loss_source)
            #source_scale_trunc = tf.cond(self.source_scale > 1.0, lambda: self.return_1(), lambda: self.source_scale)

            #self.target_scale.assign(loss_source / loss_target)
            #target_scale_trunc = tf.cond(self.target_scale > 1.0, lambda: self.return_1(), lambda: self.target_scale)

            #tf.print(target_scale_trunc * loss_source, source_scale_trunc * loss_target)

            #loss_d = self.lr_scale * (target_scale_trunc * loss_target + source_scale_trunc * loss_source)
            loss_d = loss_target + loss_source

        #tf.print(loss_d)

        grads_dis = tape.gradient(loss_d, self.discriminator.trainable_weights)

        self.optimizer_discriminator.apply_gradients(zip(grads_dis, self.discriminator.trainable_weights))
        self.optimizer_discriminator.apply_gradients(
            (grad, var)
            for (grad, var) in zip(grads_fake, self.pyrapose.trainable_weights)
            if grad is not None
        ) # to suppress warning because of not updating cls and mask
        self.optimizer_generator.apply_gradients(zip(grads_gen, self.pyrapose.trainable_weights))

        #self.optimizer_generator.apply_gradients(zip(grads_gen, self.pyrapose.trainable_weights))
        #self.optimizer_discriminator.apply_gradients(zip(grads_gen, self.model_disc.trainable_weights))

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



