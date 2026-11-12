"""Helper package for the C-MAPSS remaining-useful-life project.

Note for macOS: PyTorch (Phase 2) and XGBoost (Phase 1) ship different OpenMP
runtimes, and using both in one Python process can crash or hang. The code keeps
them apart: XGBoost is only imported inside src.models.make_model, and the Phase 2
notebooks get Phase 1's XGBoost predictions from scripts/xgb_test_predictions.py,
which runs in its own process.
"""
