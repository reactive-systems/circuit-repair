======================================================
Iterative Circuit Repair Against Formal Specifications
======================================================


.. image:: https://img.shields.io/pypi/pyversions/ml2
    :target: https://www.python.org
.. image:: https://img.shields.io/github/license/reactive-systems/ml2
    :target: https://github.com/reactive-systems/ml2/blob/main/LICENSE


This is the implementation of *Iterative Circuit Repair Against Formal Specifications* (`ICLR'23 <https://openreview.net/forum?id=SEcSahl0Ql>`_). It builds on the `ML2 library <https://github.com/reactive-systems/ml2>`_. 

ML2 is an open source Python library for machine learning research on mathematical and logical reasoning problems. The library includes the (re-)implementation of the research papers `Teaching Temporal Logics to Neural Networks <https://iclr.cc/virtual/2021/poster/3332>`_, `Neural Circuit Synthesis from Specification Patterns <https://proceedings.neurips.cc/paper/2021/file/8230bea7d54bcdf99cdfe85cb07313d5-Paper.pdf>`_ and `Iterative Circuit Repair Against Formal Specifications <https://openreview.net/forum?id=SEcSahl0Ql>`_. So far, the focus of ML2 is on propositional and linear-time temporal logic (LTL) and sequence-to-sequence models, such as the `Transformer <https://arxiv.org/abs/1706.03762>`_ and `hierarchical Transformer <https://arxiv.org/abs/2006.09265>`_. ML2 is actively developed at `CISPA Helmholtz Center for Information Security <https://cispa.de/en>`_.


Requirements
------------

- `Docker <https://www.docker.com>`_
- `Python 3.8 <https://www.python.org/dev/peps/pep-0569/>`_

Note on Docker: For data generation, evaluation, and benchmarking ML2 uses a variety of research tools (e.g. SAT solvers, model checker, and synthesis tools). For ease of use, each tool is encapsulated in a separate Docker container that is automatically pulled and launched when the tool is needed. Thus, ML2 requires Docker to be installed and running.

Installation
------------

**Before installing ML2, please note the Docker requirement.**

From Source
~~~~~~~~~~~

To install ML2 from source, clone the git repo and install with pip as follows:

.. code:: shell

    git https://github.com/reactive-systems/circuit-repair.git && \
    cd ml2 && \
    pip install .

For development pip install in editable mode and include the development dependencies as follows:

.. code:: shell

    pip install -e .[dev]

Iterative Circuit Repair Against Formal Specifications (`to appear at ICLR'23 <https://openreview.net/forum?id=SEcSahl0Ql>`_)
--------------------------------------------------------------------------------------------------------------------------------------------------------

We present a deep learning approach for repairing sequential circuits against formal specifications given in linear-time temporal logic (LTL). Given a defective circuit and its formal specification, we train Transformer models to output circuits that satisfy the corresponding specification. We propose a separated hierarchical Transformer for multimodal representation learning of the formal specification and the circuit. We introduce a data generation algorithm that enables generalization to more complex specifications and out-of-distribution datasets. In addition, our proposed repair mechanism significantly improves the automated synthesis of circuits from LTL specifications with Transformers. It improves the state-of-the-art by 6.8 percentage points on held-out instances and 11.8 percentage points on an out-of-distribution dataset from the annual reactive synthesis competition.

Datasets
~~~~~~~~

A notebook guiding through the data generation can be found in *notebooks/repair_datasets_creation.ipynb*. A notebook giving an overview over all created datasets can be found in *notebooks/datasets.ipynb*. We provide a tabular overview at `Google Sheets <https://docs.google.com/spreadsheets/d/e/2PACX-1vRshLfy0d6xFXVWOey0QTslL0cnf-DVpgnmdKsLiqAjGfYp2p0iLH_9gxGssw9bTc75PStkuoSY2TQm/pubhtml?gid=975068129&single=true>`_. 

Training
~~~~~~~~

To train a separated hierarchical Transformer with default parameters:

.. code:: shell

    python -m ml2.ltl.ltl_repair.ltl_repair_sep_hier_transformer_experiment train -n exp-repair-gen-96 -d scpa-repair-gen-96 --steps 20000 --val-freq 100 -u --tf-shuffle-buffer-size 10000

Evaluation
~~~~~~~~~~
To evaluate a model on the ``Repair`` dataset from our paper run the following command.

.. code:: shell

    python -m ml2.ltl.ltl_repair.ltl_repair_sep_hier_transformer_experiment eval -n exp-repair-gen-96-0 -u -d val --beam-sizes 16

To iteratively evaluate on the LTL synthesis problem, run the following command.

.. code:: shell

    python -m ml2.ltl.ltl_repair.ltl_repair_sep_hier_transformer_experiment pipe -n exp-repair-gen-96-0 --base-model repair-data-2 --beam-base 16 --beam-repair 16 --repeats 2 --samples 350 -d syntcomp --keep all


A notebook in *notebooks/experiments.ipynb* guides through analysis of the evaluation results.


Ablations
~~~~~~~~~

The results of training on a large selection of our diverse datasets and the results of our hyperparamter study can be found in `Google Sheets <https://docs.google.com/spreadsheets/d/e/2PACX-1vRshLfy0d6xFXVWOey0QTslL0cnf-DVpgnmdKsLiqAjGfYp2p0iLH_9gxGssw9bTc75PStkuoSY2TQm/pubhtml?gid=450169976&single=true>`_.

How to Cite
~~~~~~~~~~~

.. code:: tex

    @inproceedings{cosler_iterative_2023,
        title    = {Iterative Circuit Repair Against Formal Specifications},
        url      = {https://openreview.net/forum?id=SEcSahl0Ql},
        language = {en},
        booktitle = {International Conference on Learning Representations},
        author   = {Cosler, Matthias and Schmitt, Frederik and Hahn, Christopher and Finkbeiner, Bernd},
        year     = {2023},
        pubstate = {forthcoming}
    }
