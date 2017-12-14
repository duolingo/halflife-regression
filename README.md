# Half-Life Regression

Copyright (c) 2016 [Duolingo, Inc.](https://duolingo.com) MIT License.

Half-life regression (HLR) is a model for spaced repetition practice, with particular applications to second language acquisition. The model marries psycholinguistic theory with modern machine learning techniques, indirectly estimating the "half-life" of words (and potentially any other item or fact) in a student's long-term memory.

This repository contains a public release of the data and code used for several experiments in the following paper (which introduces HLR):

> B. Settles and B. Meeder. 2016. [A Trainable Spaced Repetition Model for Language Learning](settles.acl16.pdf).
> In _Proceedings of the Association for Computational Linguistics (ACL)_, pages 1848-1858.

When using this data set and/or software, please cite this publication. A BibTeX record is:

```
@inproceedings{settles.acl16,
    Author = {B. Settles and B. Meeder},
    Booktitle = {Proceedings of the Association for Computational Linguistics (ACL)},
    Pages = {1848--1858},
    Publisher = {ACL},
    Title = {A Trainable Spaced Repetition Model for Language Learning},
    Year = {2016}
    DOI = {10.18653/v1/P16-1174},
    URL = {http://www.aclweb.org/anthology/P16-1174}
}
```


## Software

The file ``experiment.py`` contains a Python implementation of half-life regression, as well as several baseline spaced repetition algorithms used in Section 4.1 of the paper above. It is implemented in pure Python, and we recommend using [pypy](http://pypy.org/) on large data sets for efficiency. The software creates the subfolder ``results/`` for outputting model predictions on the test partition and induced model weights for inspection.

The file ``evaluation.r`` implements an R function, ``sr_evaluate()``, which takes a prediction file from the script above and implements the three metrics we use for evaluation: mean absolute error (MAE), area under the ROC curve (AUC), and Spearman correlation for estimated half-life. Significance tests are also included.


## Data Set and Format

The data set is available on [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME) (361 MB). This is a gzipped CSV file containing the 13 million Duolingo student learning traces used in our experiments.

The columns are as follows:

* ``p_recall`` - proportion of exercises from this lesson/practice where the word/lexeme was correctly recalled
* ``timestamp`` - UNIX timestamp of the current lesson/practice
* ``delta`` - time (in seconds) since the last lesson/practice that included this word/lexeme
* ``user_id`` - student user ID who did the lesson/practice (anonymized)
* ``learning_language`` - language being learned
* ``ui_language`` - user interface language (presumably native to the student)
* ``lexeme_id`` - system ID for the lexeme tag (i.e., word)
* ``lexeme_string`` - lexeme tag (see below)
* ``history_seen`` - total times user has seen the word/lexeme prior to this lesson/practice
* ``history_correct`` - total times user has been correct for the word/lexeme prior to this lesson/practice
* ``session_seen`` - times the user saw the word/lexeme during this lesson/practice
* ``session_correct`` - times the user got the word/lexeme correct during this lesson/practice

The ``lexeme_string`` column contains a string representation of the "lexeme tag" used by Duolingo for each lesson/practice (data instance) in our experiments. It has been added for this release to facilitate future research and analysis. Only the ``lexeme_id`` column was used in our original experiments. The ``lexeme_string`` field uses the following format:

```
surface-form/lemma<pos>[<modifiers>...]
```

Where ``surface-form`` refers to the inflected form seen in (or intended for) the exercise, ``lemma`` is the uninflected root, ``pos`` is the high-level part of speech, and each of the ``modifers`` encodes a morphological component specific to the surface form (tense, gender, person, case, etc.). A few examples from Spanish:

```
bajo/bajo<pr>
blancos/blanco<adj><m><pl>
carta/carta<n><f><sg>
de/de<pr>
diario/diario<n><m><sg>
ellos/prpers<prn><tn><p3><m><pl>
es/ser<vbser><pri><p3><sg>
escribe/escribir<vblex><pri><p3><sg>
escribimos/escribir<vblex><pri><p1><pl>
lee/leer<vblex><pri><p3><sg>
lees/leer<vblex><pri><p2><sg>
leo/leer<vblex><pri><p1><sg>
libro/libro<n><m><sg>
negra/negro<adj><f><sg>
persona/persona<n><f><sg>
por/por<pr>
son/ser<vbser><pri><p3><pl>
soy/ser<vbser><pri><p1><sg>
y/y<cnjcoo>
```

Some tags contain wildcard components, written as ``<*...>``. For example, ``<*sf>`` refers to a "generic" lexeme without any specific surface form (e.g., a lexeme tag that represents _all_ conjugations of a verb: "run," "ran," "running," etc.). The ``<*numb>`` modifier subsumes both singular and plural forms of a noun (e.g., "teacher" and "teachers"). The file ``lexeme_reference.txt`` contains a reference of pos and modifier components used for lexeme tags.
