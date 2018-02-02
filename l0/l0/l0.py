import tensorflow as tf
import random

def input_fn():
    a, b, c = [], [], []
    for i in range(10000):
        aa = random.randrange(100)
        bb = random.randrange(100)
        a.append(aa)
        b.append(bb)
        c.append((aa+bb)%2)
    print('input_fn')
    return dict([('a', tf.constant(a)), ('b', tf.constant(b))]), tf.constant(c)

tf.logging.set_verbosity(tf.logging.INFO)
columns = [tf.feature_column.numeric_column('a'), tf.feature_column.numeric_column('b')]
classifier = tf.estimator.DNNClassifier(feature_columns=columns, hidden_units=[1024, 512, 256, 128], n_classes=2)
for i in range(10):
    classifier.train(input_fn=input_fn, steps=100)
metrics = classifier.evaluate(input_fn=input_fn, steps=1000)
print(metrics)
print(metrics['accuracy'])