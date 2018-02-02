import tensorflow as tf
import random

def input_fn():
    a, b, c = [], [], []
    for i in range(1000):
        a.append(random.randrange(100))
        b.append(random.randrange(100))
        c.append((a[-1] + b[-1]) % 2)
    return dict([('a', tf.constant(a)), ('b', tf.constant(b))]), tf.constant(c)

tf.logging.set_verbosity(tf.logging.INFO)
columns = [tf.feature_column.numeric_column('a'), tf.feature_column.numeric_column('b')]
classifier = tf.estimator.DNNClassifier(feature_columns=columns, hidden_units=[300], n_classes=2)
classifier.train(input_fn=lambda:input_fn(), max_steps=10000)
metrics = classifier.evaluate(input_fn=lambda:input_fn(), steps=1000)
print(metrics)
print(metrics['accuracy'])