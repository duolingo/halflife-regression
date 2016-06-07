# Half-Life Regression

Copyright (c) 2016 [Duolingo, Inc.](https://duolingo.com) MIT License.

Half-life regression (HLR) is a model for spaced repetition practice, with particular applications to second language acquisition. The model marries psycholinguistic theory with modern machine learning techniques, indirectly estimating the "half-life" of words (and potentially any other item or fact) in a student's long-term memory.

This repository contains a public release of the data and code used for several experiments in the following paper (which introduces HLR):

> B. Settles and B. Meeder. 2016. A Trainable Spaced Repetition Model for Language Learning.
> In _Proceedings of the Association for Computational Linguistics (ACL)_, to appear.

When using this data set and/or software, please cite this publication. A BibTeX record is:

```
@inproceedings{settles.acl16,
    Author = {B. Settles and B. Meeder},
    Booktitle = {Proceedings of the Association for Computational Linguistics (ACL)},
    Publisher = {ACL},
    Title = {A Trainable Spaced Repetition Model for Language Learning},
    Year = {2016}
}
```


## Software

The file ``experiment.py`` contains a Python implementation of half-life regression, as well as several baseline spaced repetition algorithms used in Section 4.1 of the paper above. It implemented in pure Python, and we recommend using [pypy](http://pypy.org/) on large datasets for efficiency. The software creates the subfolder ``results/`` which contains model predictions on the test set and induced model weights for inspection.

The file ``evaluation.r`` contains an R function, ``sr_evaluate()``, which takes a prediction file from the script above, and implements the three key metrics we use for evaluation: mean absolute error (MAE), area under the ROC curve (AUC), and Spearman correlation for estimated half-life. Significance tests are also included.


## Dataset and Format

The dataset is available here: [settles.acl16.learning_traces.13m.csv.gz](https://s3.amazonaws.com/duolingo-papers/publications/settles.acl16.learning_traces.13m.csv.gz) (361 MB).

This is a gzipped CSV file containing the 13 million Duolingo student learning traces used in our experiments. The columns are as follows:

* ``p_recall`` - proportion of exercises from this lesson/practice where the word/lexeme was correctly recalled
* ``timestamp`` - UNIX timestamp of the current lesson/practice session
* ``delta`` - time (in seconds) since the last lesson/practice session that included this word
* ``user_id`` - student user ID (anonymized)
* ``learning_language`` - language being learned
* ``ui_language`` - user interface language (presumably native to the student)
* ``lexeme_id`` - system ID for the lexeme tag (i.e., word/concept)
* ``lexeme_string`` - lexeme tag (see below)
* ``history_seen`` - total times user has seen the lexeme tag prior to this session
* ``history_correct`` - total times user has been correct for the lexeme tag prior to this session
* ``session_seen`` - times the user saw the lexeme tag during this session
* ``session_correct`` - times the user got the lexeme tag correct during this session

The ``lexeme_string`` column contains the "lexeme tag" used by the Duolingo system for each data instance at the time of these experiments. It has been added for this release to facilitate future research and analysis. Only the ``lexeme_id`` column was used in our original experiments. The ``lexeme_string`` field uses the following format:

```
surface-form/lemma<pos>[<modifiers>...]
```

Where ``surface-form`` refers to the inflected form seen in (or intended for) the exercise, ``lemma`` is the uninflected root, ``pos`` is the high-level part of speech, and each of the ``modifers`` encodes a morphological component specific to the surface form (tense, gender, person, case, etc.).

Some tags contain wildcard components, written ``<*...>``. For example, ``<*sf>`` refers to a "generic" lexeme without any specific surface form (e.g., a lexeme tag that represents _all_ conjugations of a verb: "run," "ran," "running," etc.), or ``<*numb>`` (e.g., both singular and plural forms of a noun: "teacher" and "teachers"). The file ``lexeme_reference.txt`` contains a reference of pos and modifier components used for lexeme tags.
