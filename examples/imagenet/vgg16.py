import chainer
import chainer.functions as F
import chainer.links as L


class VGG16(chainer.Chain):
    insize = 224
    
    def __init__(self):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, pad=1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, pad=1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, pad=1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, pad=1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, pad=1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, pad=1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, pad=1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, pad=1)
            self.conv5_1 = L.Convolution2D(512, 512, 3, pad=1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, pad=1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, pad=1)
            self.fc6 = L.Linear(25088, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

    def __call__(self, x, t):
        h1_1 = F.relu(self.conv1_1(x))
        h1_2 = F.relu(self.conv1_2(h1_1))
        h1 = F.max_pooling_2d(h1_2, 2, stride=2)
        h2_1 = F.relu(self.conv2_1(h1))
        h2_2 = F.relu(self.conv2_2(h2_1))
        h2 = F.max_pooling_2d(h2_2, 2, stride=2)
        h3_1 = F.relu(self.conv3_1(h2))
        h3_2 = F.relu(self.conv3_2(h3_1))
        h3_3 = F.relu(self.conv3_3(h3_2))
        h3 = F.max_pooling_2d(h3_3, 2, stride=2)
        h4_1 = F.relu(self.conv4_1(h3))
        h4_2 = F.relu(self.conv4_2(h4_1))
        h4_3 = F.relu(self.conv4_3(h4_2))
        h4 = F.max_pooling_2d(h4_3, 2, stride=2)
        h5_1 = F.relu(self.conv5_1(h4))
        h5_2 = F.relu(self.conv5_2(h5_1))
        h5_3 = F.relu(self.conv5_3(h5_2))
        h5 = F.max_pooling_2d(h5_3, 2, stride=2)
        h6 = F.dropout(F.relu(self.fc6(h5)), ratio=0.5)
        h7 = F.dropout(F.relu(self.fc7(h6)), ratio=0.5)
        h8 = self.fc8(h7)

        loss = F.softmax_cross_entropy(h8, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h8, t)}, self)
    return loss
