# This is based on the following file:
# https://github.com/scikit-learn/scikit-learn/blob/2bd87f6ed8d36f89cab73552665e4e59e892262b/examples/model_selection/plot_precision_recall.py
# The original file is licensed under the New BSD License with the following
# copyright notice:
# Copyright (c) 2007–2019 The scikit-learn developers.
# All rights reserved.

"""
================
Precision-Recall
================

Example of Precision-Recall metric to evaluate classifier output quality.

Precision-Recall is a useful measure of success of prediction when the
classes are very imbalanced. In information retrieval, precision is a
measure of result relevancy, while recall is a measure of how many truly
relevant results are returned.

The precision-recall curve shows the tradeoff between precision and
recall for different threshold. A high area under the curve represents
both high recall and high precision, where high precision relates to a
low false positive rate, and high recall relates to a low false negative
rate. High scores for both show that the classifier is returning accurate
results (high precision), as well as returning a majority of all positive
results (high recall).

A system with high recall but low precision returns many results, but most of
its predicted labels are incorrect when compared to the training labels. A
system with high precision but low recall is just the opposite, returning very
few results, but most of its predicted labels are correct when compared to the
training labels. An ideal system with high precision and high recall will
return many results, with all results labeled correctly.

Precision (:math:`P`) is defined as the number of true positives (:math:`T_p`)
over the number of true positives plus the number of false positives
(:math:`F_p`).

:math:`P = \\frac{T_p}{T_p+F_p}`

Recall (:math:`R`) is defined as the number of true positives (:math:`T_p`)
over the number of true positives plus the number of false negatives
(:math:`F_n`).

:math:`R = \\frac{T_p}{T_p + F_n}`

These quantities are also related to the (:math:`F_1`) score, which is defined
as the harmonic mean of precision and recall.

:math:`F1 = 2\\frac{P \\times R}{P+R}`

Note that the precision may not decrease with recall. The
definition of precision (:math:`\\frac{T_p}{T_p + F_p}`) shows that lowering
the threshold of a classifier may increase the denominator, by increasing the
number of results returned. If the threshold was previously set too high, the
new results may all be true positives, which will increase precision. If the
previous threshold was about right or too low, further lowering the threshold
will introduce false positives, decreasing precision.

Recall is defined as :math:`\\frac{T_p}{T_p+F_n}`, where :math:`T_p+F_n` does
not depend on the classifier threshold. This means that lowering the classifier
threshold may increase recall, by increasing the number of true positive
results. It is also possible that lowering the threshold may leave recall
unchanged, while the precision fluctuates.

The relationship between recall and precision can be observed in the
stairstep area of the plot - at the edges of these steps a small change
in the threshold considerably reduces precision, with only a minor gain in
recall.

**Average precision** (AP) summarizes such a plot as the weighted mean of
precisions achieved at each threshold, with the increase in recall from the
previous threshold used as the weight:

:math:`\\text{AP} = \\sum_n (R_n - R_{n-1}) P_n`

where :math:`P_n` and :math:`R_n` are the precision and recall at the
nth threshold. A pair :math:`(R_k, P_k)` is referred to as an
*operating point*.

AP and the trapezoidal area under the operating points
(:func:`sklearn.metrics.auc`) are common ways to summarize a precision-recall
curve that lead to different results. Read more in the
:ref:`User Guide <precision_recall_f_measure_metrics>`.

Precision-recall curves are typically used in binary classification to study
the output of a classifier. In order to extend the precision-recall curve and
average precision to multi-class or multi-label classification, it is necessary
to binarize the output. One curve can be drawn per label, but one can also draw
a precision-recall curve by considering each element of the label indicator
matrix as a binary prediction (micro-averaging).

.. note::

    See also :func:`sklearn.metrics.average_precision_score`,
             :func:`sklearn.metrics.recall_score`,
             :func:`sklearn.metrics.precision_score`,
             :func:`sklearn.metrics.f1_score`
"""
from __future__ import print_function

###############################################################################
# In binary classification settings
# --------------------------------------------------------
#
# Create simple data
# ..................
#
# Try to differentiate the two first classes of the iris data
import numpy as np
import pickle

with open('walk_scores.p', 'rb') as f:
    walk_scores = pickle.load(f)

with open('walk_labels.p', 'rb') as f:
    walk_labels = pickle.load(f)

with open('dont_walk_scores.p', 'rb') as f:
    dont_walk_scores = pickle.load(f)

with open('dont_walk_labels.p', 'rb') as f:
    dont_walk_labels = pickle.load(f)

###############################################################################
# Compute the average precision score
# ...................................
from sklearn.metrics import average_precision_score
walk_average_precision = average_precision_score(walk_labels, walk_scores)

print('Walk average precision-recall score: {0:0.2f}'.format(
      walk_average_precision))

dont_walk_average_precision = average_precision_score(dont_walk_labels, dont_walk_scores)

print('Don\'t walk average precision-recall score: {0:0.2f}'.format(
      dont_walk_average_precision))

###############################################################################
# Plot the Precision-Recall curve
# ................................
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

walk_precision, walk_recall, _ = precision_recall_curve(walk_labels, walk_scores)
dont_walk_precision, dont_walk_recall, _ = precision_recall_curve(
    dont_walk_labels, dont_walk_scores)

plt.plot(dont_walk_recall, dont_walk_precision, color='green', lw=2,
         label='Precision-recall for Walk, AP={0:0.2f}'.format(
             walk_average_precision))
plt.plot(walk_recall, walk_precision, color='red', lw=2,
         label='Precision-recall for Don\'t walk, '
         'AP={0:0.2f}'.format(
             dont_walk_average_precision))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
# plt.title('Precision-Recall curve for Faster R-CNN')
plt.title('Precision-Recall curve for SSD')
plt.legend(loc='lower left')

plt.show()
