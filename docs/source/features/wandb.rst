Experiment tracking with Weights & Biases (W&B) in SMLE
-------------------------------------------------------

A step by step guide on how to integrate `Weights & Biases <https://wandb.ai>`_
(W&B) into an SMLE-based project in order to track machine learning experiments
(metrics, hyperparameters, models, and training logs).

.. note::
    This guide assumes that you have an active W&B account at https://wandb.ai.


1. Create a W&B account and obtain your API key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you can log experiments from SMLE to W&B, you must create a W&B account and obtain a personal API key associated with that account.
This key is used to authenticate all requests from your code to the W&B backend.

#. Loggin in `https://wandb.ai <https://wandb.ai>`_ and open your user *Settings* page from the account menu.
#. Locate the **API Keys** section and create or copy your personal API key, which will be required to authenticate the W&B client from your code.


⚠️ Security & WandB Configuration
"""""""""""""""""""""""""""""""""

.. warning::
    When using the ``wandb`` section for remote logging, the API key is read from ``smle.yaml``. To avoid exposing credentials, do not commit ``smle.yaml`` or log files with real keys to any public repository.

    It is a good practice to:
        * Add ``smle.yaml`` and ``*.log`` to ``.gitignore``.
        * Remove the ``wandb`` section entirely if remote logging is not required.


2. Configuring SMLE: the ``smle.yaml`` file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SMLE uses a ``smle.yaml`` file to centralize project configuration
(training, logging, etc.). To enable W&B, add a dedicated section.

Minimal example of ``smle.yaml`` with W&B support:

.. code-block:: yaml

   wandb:
        entity: your_wandb_account
        key: your_wandb_key


3. Running an experiment
^^^^^^^^^^^^^^^^^^^^^^^^

Once the ``smle.yaml`` file has been configured, including the ``wandb`` section, you can start an experiment by running your SMLE entrypoint script as a standard Python program:

.. code-block:: bash

   python main.py

During execution, the script reads the configuration from ``smle.yaml``, initializes the W&B client, and sends configuration data and training metrics to your W&B project.
When the run completes, you can open `https://wandb.ai <https://wandb.ai>`_, navigate to the configured project, and inspect the dashboards with loss and accuracy curves, the stored experiment configurations, and any saved files such as models and logs.



