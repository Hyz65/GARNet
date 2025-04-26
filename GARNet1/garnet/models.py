from garnet.layers import *
from garnet.metrics import *

flags = tf.flags
FLAGS = flags.FLAGS



class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.hid = None

        self.loss = 0
        self.accuracy = 0
        self.opt_op_gen = None
        self.opt_op_disc = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        self._build()

    def train_step1(self):
        pass
       
    def predict(self):
        pass

    def hidd(self):
        pass

import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Discriminator, self).__init__()
        self.hidden_layer1 = tf.keras.layers.Dense(hidden_dim1, activation='relu')
        self.hidden_layer2 = tf.keras.layers.Dense(hidden_dim2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def build(self, input_shape):
        # 显式设置输入形状，触发权重初始化
        self.hidden_layer1.build(input_shape)
        self.hidden_layer2.build((input_shape[0], self.hidden_layer1.units))
        self.output_layer.build((input_shape[0], self.hidden_layer2.units))

    def call(self, inputs):
        hidden1 = self.hidden_layer1(inputs)
        hidden2 = self.hidden_layer2(hidden1)
        output = self.output_layer(hidden2)
        return output



class Discriminator(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Discriminator, self).__init__()
        self.hidden_layer1 = tf.keras.layers.Dense(hidden_dim1, activation='relu')
        self.hidden_layer2 = tf.keras.layers.Dense(hidden_dim2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def build(self, input_shape):
        # 显式设置输入形状，触发权重初始化
        self.hidden_layer1.build(input_shape)
        self.hidden_layer2.build((input_shape[0], self.hidden_layer1.units))
        self.output_layer.build((input_shape[0], self.hidden_layer2.units))

    def call(self, inputs):
        hidden1 = self.hidden_layer1(inputs)
        hidden2 = self.hidden_layer2(hidden1)
        output = self.output_layer(hidden2)
        return output

class FocalLoss(tf.keras.losses.Loss):
        def __init__(self, alpha=0.25, gamma=2, name="focal_loss"):
            super(FocalLoss, self).__init__(name=name)
            self.alpha = alpha
            self.gamma = gamma

        def call(self, y_true, y_pred):
            cross_entropy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            loss = self.alpha * tf.pow(1 - y_pred, self.gamma) * cross_entropy_loss
            return tf.reduce_mean(loss)


class GARNet(tf.keras.Model):
    def __init__(self, input_dim, size_gene, latent_factor_num, hidden1, hidden2, dropout, learning_rate, discriminator_learning_rate, adj, **kwargs):
        super(GARNet, self).__init__(**kwargs)
        
        self.size_gene = size_gene
        self.input_dim = input_dim
        self.latent_factor_num = latent_factor_num
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout
        self.adj = adj  # 将 adj 直接作为实例变量保存
        self.loss = None
        self.accuracy = None

        # 优化器
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_learning_rate)

        # 定义生成器的层
        self.encoder1 = Encoder(input_dim=self.input_dim, output_dim=self.hidden1, gene_size=self.size_gene,
                        adj=self.adj, dropout=0., featureless=False)
        self.encoder2 = Encoder(input_dim=self.hidden1, output_dim=self.latent_factor_num, gene_size=self.size_gene,
                        adj=self.adj, dropout=self.dropout, featureless=False, act=lambda x: x)
        self.decoder = Decoder(size1=self.size_gene,latent_factor_num=self.latent_factor_num)

        # 鉴别器
        self.discriminator = Discriminator(input_dim=self.latent_factor_num, hidden_dim1=self.hidden1,
                                           hidden_dim2=self.hidden2, output_dim=1)

    
    def call(self, inputs, adjacency_matrix=None,training=False):
        """前向传播"""
        if adjacency_matrix is None:
            adjacency_matrix = self.adj  # 如果没有传入，则使用初始化时的 adj

        x1 = self.encoder1(inputs)
        x1 = self.encoder2(x1)
        x2 = self.decoder(x1)
        return x1, x2  # 直接返回局部张量，不再存储为实例变量

    def compute_discriminator_loss(self, real_distribution, generated_data):
        """计算鉴别器的损失"""
        real_output = self.discriminator(real_distribution)  # 真实分布的输出
        fake_output = self.discriminator(generated_data)  # 生成分布的输出

        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        disc_loss_real = bce_loss(tf.ones_like(real_output), real_output)
        disc_loss_fake = bce_loss(tf.zeros_like(fake_output), fake_output)

        return disc_loss_real + disc_loss_fake

    def compute_generator_loss(self, generated_data):
        """计算生成器的损失"""
        fake_output = self.discriminator(generated_data)
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return bce_loss(tf.ones_like(fake_output), fake_output)
    
    def update_loss_and_accuracy(self, labels, pred_data):
        """更新模型的损失和准确率"""
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss = bce_loss(labels, pred_data,sample_weight=0.1)


        predictions = tf.argmax(pred_data, axis=1)
        correct_prediction = tf.equal(predictions, tf.argmax(labels, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    

    
    @tf.function 
    def train_step1(self, features, labels, real_distribution):
        """训练一步"""
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 前向传播，传递必需的参数
            generated_data, pred_data = self.call(features)  # 使用返回值
            self.update_loss_and_accuracy(labels, pred_data)
            
            #focal_loss = FocalLoss()
            #self.loss = focal_loss(labels, pred_data)


            # 计算生成器和鉴别器损失
            gen_loss = self.compute_generator_loss(generated_data)
            disc_loss = self.compute_discriminator_loss(real_distribution, generated_data)

            # 梯度惩罚项
            alpha = tf.random.uniform(shape=[features.shape[0], 1], minval=0., maxval=1.)
            interpolated = alpha * real_distribution + (1 - alpha) * generated_data

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                interpolated_output = self.discriminator(interpolated)

            gradients = gp_tape.gradient(interpolated_output, [interpolated])[0]
            gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
            gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
            lambda_gp = 1
            disc_loss += lambda_gp * gradient_penalty

        # 计算梯度并更新参数
        gradients_gen = gen_tape.gradient(gen_loss, self.trainable_variables)
        gradients_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_gen = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_gen]
        gradients_disc = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_disc]
        self.generator_optimizer.apply_gradients(zip(gradients_gen, self.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_disc, self.discriminator.trainable_variables))
        

        return gen_loss, disc_loss, self.loss,self.accuracy
        #return gen_loss, disc_loss,self.accuracy
        #return gen_loss, disc_loss
    @tf.function 
    def train_step2(self, features, real_distribution):
        """训练一步"""
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 前向传播，传递必需的参数
            generated_data, pred_data = self.call(features)  # 使用返回值
            #self.update_loss_and_accuracy(labels, pred_data)
            
            #focal_loss = FocalLoss()
            #self.loss = focal_loss(labels, pred_data)


            # 计算生成器和鉴别器损失
            gen_loss = self.compute_generator_loss(generated_data)
            disc_loss = self.compute_discriminator_loss(real_distribution, generated_data)

            # 梯度惩罚项
            alpha = tf.random.uniform(shape=[features.shape[0], 1], minval=0., maxval=1.)
            interpolated = alpha * real_distribution + (1 - alpha) * generated_data

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                interpolated_output = self.discriminator(interpolated)

            gradients = gp_tape.gradient(interpolated_output, [interpolated])[0]
            gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
            gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
            lambda_gp = 1
            disc_loss += lambda_gp * gradient_penalty

        # 计算梯度并更新参数
        gradients_gen = gen_tape.gradient(gen_loss, self.trainable_variables)
        gradients_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_gen = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_gen]
        gradients_disc = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_disc]
        self.generator_optimizer.apply_gradients(zip(gradients_gen, self.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_disc, self.discriminator.trainable_variables))
        

        return gen_loss, disc_loss
        #return gen_loss, disc_loss,self.accuracy
        #return gen_loss, disc_loss
    def predict(self, inputs):
        _, decoded_output = self.call(inputs)  # 使用 _ 忽略 x1
        return decoded_output

