#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from tqdm import tqdm
from bert4keras.backend import K,keras,search_layer
from metric_utils import *


maxlen = 256
epochs = 6
batch_size = 4
learning_rate = 2e-5
categories = set()

# 模型保存路径
ckpt_save_path = './model_result/bert-GP.weights'

# bert配置
config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'

# # roberta配置
# config_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

# # wobert配置
# config_path = './chinese_wobert_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = './chinese_wobert_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = './chinese_wobert_L-12_H-768_A-12/vocab.txt'

# # simBERT配置
# config_path = './chinese_simbert_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = './chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = './chinese_simbert_L-12_H-768_A-12/vocab.txt'

def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D

# 标注数据
train_data = load_data('./data/example.train')
valid_data = load_data('./data/example.dev')
test_data = load_data('./data/example.test')
categories = list(sorted(categories))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class SetLearningRate:
    """
    层的一个包装，用来设置当前层的学习率
    """

    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb  # 学习率比例
        self.is_ada = is_ada  # 是否自适应学习率优化器

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma',
                    'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb  # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb ** 0.5  # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb)  # 更改初始化
                setattr(self.layer, key, weight * lamb)  # 按比例替换
        return self.layer(inputs)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))

def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    # return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)
    p = K.sum(y_true * y_pred) / K.sum(y_pred)
    r = K.sum(y_true * y_pred) / K.sum(y_true)
    return 2*p*r/(p+r)

def adversarial_training(model, embedding_name, epsilon=1):
    """
    给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):
        # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs
    model.train_function = train_function  # 覆盖原训练函数


# bert-bilstm-gp
def build_model_1(config_path, checkpoint_path, categories_num, learning_rate):
    # categories_num = len(categories)
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='bert',
        return_keras_model=False
    )
    x = bert.model.output  # [batch_size, seq_length, 768]
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            128,
            kernel_initializer='he_normal',
            return_sequences=True
        )
    )(x)  # [batch_size, seq_length, lstm_units * 2]

    x = keras.layers.TimeDistributed(
        keras.layers.Dropout(0.1)
    )(x)

    output = GlobalPointer(categories_num, 64)(x)  # [bs,heads,maxlen,maxlen]
    model = Model(bert.input, output)
    model.compile(
        loss=global_pointer_crossentropy,
        optimizer=Adam(learning_rate),
        metrics=[global_pointer_f1_score]
    )

    return model

# bert-bilstm-gp 分层学习率等优化
def build_model_2(config_path, checkpoint_path, categories_num, learning_rate):
    # categories_num = len(categories)
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='bert',
        return_keras_model=False
    )
    x = bert.model.output  # [batch_size, seq_length, 768]

    lstm = SetLearningRate(
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                128,
                kernel_initializer='he_normal',
                return_sequences=True
            )
        ),
        500,
        True
    )(x)  # [batch_size, seq_length, lstm_units * 2]

    x = keras.layers.concatenate(
        [lstm, x],
        axis=-1
    )  # [batch_size, seq_length, lstm_units * 2 + 768]

    x = keras.layers.TimeDistributed(
        keras.layers.Dropout(0.1)
    )(x)  # [batch_size, seq_length, lstm_units * 2 + 768]

    x = SetLearningRate(
        keras.layers.TimeDistributed(
            keras.layers.Dense(
                13,
                activation='relu',
                kernel_initializer='he_normal',
            )
        ),
        500,
        True
    )(x)  # [batch_size, seq_length, num_labels]

    output = GlobalPointer(categories_num, 64)(x)  # [bs,heads,maxlen,maxlen]
    model = Model(bert.input, output)
    model.compile(
        loss=global_pointer_crossentropy,
        optimizer=Adam(learning_rate),
        metrics=[global_pointer_f1_score]
    )

    return model

# model = build_model_1(config_path, checkpoint_path, len(categories), learning_rate)
# model = build_model_2(config_path, checkpoint_path, len(categories), learning_rate)
# adversarial_training(model,'Embedding-Token',0.5)

model = build_transformer_model(config_path, checkpoint_path)
output = GlobalPointer(len(categories), 64)(model.output)
model = Model(model.input, output)
model.compile(
    loss=global_pointer_crossentropy,
    optimizer=Adam(learning_rate),
    metrics=[global_pointer_f1_score]
)

class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
        return entities
NER = NamedEntityRecognizer()

def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(ckpt_save_path)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


if __name__ == '__main__':

    # evaluator = Evaluator()
    # train_generator = data_generator(train_data, batch_size)
    #
    # model.fit(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=epochs,
    #     callbacks=[evaluator]
    # )

    # 评估
    model.load_weights(ckpt_save_path)
    true_laebl, pred_label = [], []
    for i, d in enumerate(valid_data):
        p = [(e[2], e[0], e[1]) for e in NER.recognize(d[0])]
        t = [(e[2], e[0], e[1]) for e in d[1:]]

        true_laebl.extend(t)
        pred_label.extend(p)
    print(classification_report(true_laebl, pred_label, digits=4))

else:

    model.load_weights(ckpt_save_path)



