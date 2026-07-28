"""Backtest pipeline package (DB-first).

The runnable surface is ``experiment.run_experiment`` + ``experiment.ExperimentConfig``
(loaded from the DB via ``ExperimentConfig.from_template_id_async``), driven by
``stock_ml/scripts/run_template.py``. Import those directly, e.g.::

    from src.pipeline.experiment import run_experiment, ExperimentConfig

Nothing is re-exported here so that importing the package never force-loads modules
a given entry point does not use.
"""
