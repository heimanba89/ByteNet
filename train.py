# -*- coding: utf-8 -*-
import sugartensor as tf
from data import TedTrans
from data import ComTrans
import pdb

__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # batch size
latent_dim = 400   # hidden layer dimension
num_dim = 128      # latent dimension
num_blocks = 3     # dilated blocks

#
# inputs
#

# ComTrans parallel corpus input tensor ( with QueueRunner )
data = TedTrans(batch_size=batch_size)

# source, target sentence
x, y = data.mfcc, data.target
voca_size = data.voca_size

# make embedding matrix for target
emb_y = tf.sg_emb(name='emb_y', voca_size=voca_size, dim=latent_dim)

# shift target for training source
y_src = tf.concat(1, [tf.zeros((batch_size, 1), tf.sg_intx), y[:, :-1]])

# residual block
@tf.sg_sugar_func
def sg_res_block(tensor, opt):
    # default rate
    opt += tf.sg_opt(size=3, rate=1, causal=False)

    # input dimension
    in_dim = tensor.get_shape().as_list()[-1]

    # reduce dimension
    input_ = (tensor
              .sg_bypass(act='relu', bn=(not opt.causal), ln=opt.causal)
              .sg_conv1d(size=1, dim=in_dim/2, act='relu', bn=(not opt.causal), ln=opt.causal))

    # 1xk conv dilated
    out = input_.sg_aconv1d(size=opt.size, rate=opt.rate, causal=opt.causal, act='relu', bn=(not opt.causal), ln=opt.causal)

    # dimension recover and residual connection
    out = out.sg_conv1d(size=1, dim=in_dim) + tensor

    return out

# inject residual multiplicative block
tf.sg_inject_func(sg_res_block)

# expand dimension
#enc = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True)
enc = x;

# loop dilated conv block
for i in range(num_blocks):
    enc = (enc
           .sg_res_block(size=5, rate=1)
           .sg_res_block(size=5, rate=2)
           .sg_res_block(size=5, rate=4)
           .sg_res_block(size=5, rate=8)
           .sg_res_block(size=5, rate=16))

# pooling before passed to decodering
enc = tf.expand_dims(enc, 1);
enc = tf.nn.max_pool(enc, [1, 1, 10, 1], [1, 1, 10, 1], 'VALID')
enc = tf.squeeze(enc, [1])

# zero padding encoder output to have the size equal to self.max_len
enc = tf.transpose(enc, perm=[1, 2, 0])
enc = tf.image.resize_image_with_crop_or_pad(enc, data.max_len, 20)
enc = tf.transpose(enc, perm=[2, 0 ,1])

# concat merge target source
enc = enc.sg_concat(target=y_src.sg_lookup(emb=emb_y))

#
# decode graph ( causal convolution )
#

# loop dilated causal conv block
dec = enc
for i in range(num_blocks):
    dec = (dec
           .sg_res_block(size=3, rate=1, causal=True)
           .sg_res_block(size=3, rate=2, causal=True)
           .sg_res_block(size=3, rate=4, causal=True)
           .sg_res_block(size=3, rate=8, causal=True))

# final fully convolution layer for softmax
dec = dec.sg_conv1d(size=1, dim=data.voca_size)

# cross entropy loss with logit and mask
loss = dec.sg_ce(target=y, mask=True)


# train
tf.sg_train(log_interval=30, lr=0.00001, loss=loss,
            ep_size=data.num_batch, max_ep=20, early_stop=False)

