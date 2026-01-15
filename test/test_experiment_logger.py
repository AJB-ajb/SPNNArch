"""
Unit tests for the ExpLogger class.
Tests core functionality: initialization, data/model/figure saving, loading, and prefix filtering.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock

from masterthesis.experiment_logger import ExpLogger


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        return self.linear(x)


class TestExpLogger(unittest.TestCase):
    """Test cases for ExpLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        self.test_data = {"key1": "value1", "key2": [1, 2, 3]}
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    # Basic Initialization Tests
    def test_initialization_defaults(self):
        """Test default initialization."""
        logger = ExpLogger("test_exp", base_path=self.base_path)
        
        self.assertEqual(logger.exp_name, "test_exp")
        self.assertFalse(logger.tmp)
        self.assertTrue(logger.exp_id.startswith("run_"))
        self.assertTrue(logger.experiment_folder().exists())

    def test_initialization_custom_prefix(self):
        """Test initialization with custom prefix."""
        logger = ExpLogger("test_exp", prefix="large", base_path=self.base_path)
        self.assertTrue(logger.exp_id.startswith("large_"))

    def test_initialization_tmp_mode(self):
        """Test temporary mode initialization."""
        logger = ExpLogger("test_exp", tmp=True, base_path=self.base_path)
        
        self.assertTrue(logger.tmp)
        # In tmp mode, exp_id should end with _tmp
        self.assertTrue(logger.exp_id.endswith("_tmp"))
        # Even in tmp mode, experiment folder is created on disk
        self.assertTrue(logger.experiment_folder().exists())

    def test_initialization_custom_exp_id(self):
        """Test initialization with custom experiment ID."""
        logger = ExpLogger("test_exp", exp_id="custom_id", base_path=self.base_path)
        self.assertEqual(logger.exp_id, "custom_id")

    # Data Save/Load Tests
    def test_data_save_load_disk(self):
        """Test data saving and loading to/from disk."""
        logger = ExpLogger("test_exp", base_path=self.base_path)
        
        logger.save_data(self.test_data, "test.pkl")
        loaded_data = logger.load_data("test.pkl")
        
        self.assertEqual(loaded_data, self.test_data)
        self.assertTrue((logger.experiment_folder() / "test.pkl").exists())

    def test_data_save_load_memory(self):
        """Test data saving and loading in tmp mode (still uses disk)."""
        logger = ExpLogger("test_exp", tmp=True, base_path=self.base_path)
        
        logger.save_data(self.test_data, "test.pkl")
        loaded_data = logger.load_data("test.pkl")
        
        self.assertEqual(loaded_data, self.test_data)
        # Even in tmp mode, data is saved to disk
        self.assertTrue((logger.experiment_folder() / "test.pkl").exists())

    def test_load_nonexistent_data(self):
        """Test loading nonexistent data returns None."""
        logger = ExpLogger("test_exp", base_path=self.base_path)
        result = logger.load_data("nonexistent.pkl")
        self.assertIsNone(result)

    # Model Save/Load Tests
    def test_single_model_save_load(self):
        """Test single model saving and loading."""
        logger = ExpLogger("test_exp", base_path=self.base_path)
        
        model = SimpleModel()
        original_weight = model.linear.weight.data.clone()
        
        logger.save_model(model, "model.pt")
        loaded_model = logger.load_model(SimpleModel, "model.pt")
        
        self.assertIsNotNone(loaded_model)
        torch.testing.assert_close(loaded_model.linear.weight.data, original_weight)

    def test_multiple_models_save_load(self):
        """Test multiple models saving and loading."""
        logger = ExpLogger("test_exp", base_path=self.base_path)
        
        models = {"model1": SimpleModel(), "model2": SimpleModel()}
        logger.save_models(models, "models.pt")
        
        # Test with single class
        loaded_models = logger.load_models(SimpleModel, "models.pt")
        self.assertEqual(len(loaded_models), 2)
        self.assertIn("model1", loaded_models)
        self.assertIn("model2", loaded_models)
        
        # Test with dict of classes
        model_classes = {"model1": SimpleModel, "model2": SimpleModel}
        loaded_models2 = logger.load_models(model_classes, "models.pt")
        self.assertEqual(len(loaded_models2), 2)

    def test_model_memory_mode(self):
        """Test model operations in tmp mode (still uses disk)."""
        logger = ExpLogger("test_exp", tmp=True, base_path=self.base_path)
        
        model = SimpleModel()
        logger.save_model(model, "model.pt")
        loaded_model = logger.load_model(SimpleModel, "model.pt")
        
        self.assertIsNotNone(loaded_model)
        # Even in tmp mode, model is saved to disk
        self.assertTrue((logger.experiment_folder() / "model.pt").exists())

    def test_save_load_models_cfgs_new_with_icfg(self):
        """Test new ICfg-based model saving and loading."""
        from masterthesis.icfg import ICfg
        from masterthesis.stacked_torch_modules import StackedCoupledLinearModel
        import torch.nn as nn
        
        logger = ExpLogger("test_exp", base_path=self.base_path)
        
        # Create ICfg configurations
        model_cfg = ICfg(
            cls=StackedCoupledLinearModel,
            n_instances=2,
            n_features=3,
            n_hidden=4,
            activation=nn.ReLU(),
            bias=True
        )
        
        # Create and save models
        model1 = model_cfg.instantiate()
        model2 = model_cfg.instantiate()
        models = {"model1": model1, "model2": model2}
        cfgs = {"model1": model_cfg, "model2": model_cfg}
        
        # Save using new method
        logger.save_models_cfgs_new(models, cfgs)
        
        # Load using new method
        loaded_models, loaded_cfgs = logger.load_models_cfgs_new()
        
        # Verify
        self.assertEqual(len(loaded_models), 2)
        self.assertIn("model1", loaded_models)
        self.assertIn("model2", loaded_models)
        self.assertIsInstance(loaded_models["model1"], StackedCoupledLinearModel)
        
        # Verify cfgs are also returned
        self.assertEqual(len(loaded_cfgs), 2)
        self.assertIn("model1", loaded_cfgs)
        self.assertIn("model2", loaded_cfgs)
        
        # Test weight preservation
        torch.testing.assert_close(
            loaded_models["model1"].layers['L_down'].weight.data,
            model1.layers['L_down'].weight.data
        )

    def test_load_models_cfgs_new_memory_mode(self):
        """Test ICfg loading in memory mode."""
        from masterthesis.icfg import ICfg
        from masterthesis.stacked_torch_modules import StackedCoupledLinearModel
        import torch.nn as nn
        
        logger = ExpLogger("test_exp", tmp=True, base_path=self.base_path)
        
        # Create config and model
        cfg = ICfg(
            cls=StackedCoupledLinearModel,
            n_instances=1, n_features=2, n_hidden=3,
            activation=nn.ReLU(), bias=False
        )
        
        model = cfg.instantiate()
        
        # Save and load
        logger.save_models_cfgs_new({"test": model}, {"test": cfg})
        loaded, loaded_cfgs = logger.load_models_cfgs_new()
        
        self.assertEqual(len(loaded), 1)
        self.assertIsInstance(loaded["test"], StackedCoupledLinearModel)
        self.assertEqual(len(loaded_cfgs), 1)
        self.assertIn("test", loaded_cfgs)

    def test_load_models_cfgs_new_fallback_to_saved_cfg(self):
        """Test loading without providing configs (use saved ones)."""
        from masterthesis.icfg import ICfg
        from masterthesis.stacked_torch_modules import StackedCoupledLinearModel
        import torch.nn as nn
        
        logger = ExpLogger("test_exp", base_path=self.base_path)
        
        cfg = ICfg(
            cls=StackedCoupledLinearModel,
            n_instances=1, n_features=2, n_hidden=2,
            activation=nn.ReLU(), bias=True
        )
        
        model = cfg.instantiate()
        logger.save_models_cfgs_new({"model": model}, {"model": cfg})
        
        # Load without providing configs
        loaded, loaded_cfgs = logger.load_models_cfgs_new()
        
        self.assertEqual(len(loaded), 1)
        self.assertIn("model", loaded)
        self.assertEqual(len(loaded_cfgs), 1)
        self.assertIn("model", loaded_cfgs)

    # Figure Save Tests
    def test_matplotlib_figure_save(self):
        """Test matplotlib figure saving."""
        logger = ExpLogger("test_exp", base_path=self.base_path)
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        logger.save_figure(fig, "test_fig")
        
        # Check file exists for disk mode
        fig_file = logger.experiment_folder() / "test_fig.svg"
        self.assertTrue(fig_file.exists())
        plt.close(fig)

    def test_figure_memory_mode(self):
        """Test figure saving in tmp mode (still uses disk)."""
        logger = ExpLogger("test_exp", tmp=True, base_path=self.base_path)
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        logger.save_figure(fig, "test_fig")
        
        # Even in tmp mode, figures are saved to disk
        fig_file = logger.experiment_folder() / "test_fig.svg"
        self.assertTrue(fig_file.exists())
        plt.close(fig)

    def test_unknown_figure_type(self):
        """Test handling of unknown figure types."""
        logger = ExpLogger("test_exp", base_path=self.base_path)
        
        unknown_figure = {"type": "unknown", "data": [1, 2, 3]}
        logger.save_figure(unknown_figure, "unknown_fig")  # Should not crash

    # Load Logger Tests
    def test_load_logger_basic(self):
        """Test basic load_logger functionality."""
        # Create original logger
        original_logger = ExpLogger("test_load_exp", base_path=self.base_path)
        original_logger.save_data(self.test_data, "test.pkl")
        original_exp_id = original_logger.exp_id
        
        # Load logger
        loaded_logger = ExpLogger.load_logger("test_load_exp", base_path=self.base_path)
        
        self.assertEqual(loaded_logger.exp_name, "test_load_exp")
        self.assertEqual(loaded_logger.exp_id, original_exp_id)
        
        # Verify data can be loaded
        loaded_data = loaded_logger.load_data("test.pkl")
        self.assertEqual(loaded_data, self.test_data)

    def test_load_logger_with_prefix_filter(self):
        """Test load_logger with prefix filtering."""
        import time
        
        # Create loggers with different prefixes
        logger1 = ExpLogger("prefix_test", prefix="small", base_path=self.base_path)
        logger1.save_data({"type": "small"}, "data.pkl")
        
        time.sleep(1.1)  # Ensure different timestamps (must be > 1 second due to timestamp precision)
        
        logger2 = ExpLogger("prefix_test", prefix="large", base_path=self.base_path)
        logger2.save_data({"type": "large"}, "data.pkl")
        
        # Load without prefix (should get latest - large)
        loaded_any = ExpLogger.load_logger("prefix_test", base_path=self.base_path)
        data_any = loaded_any.load_data("data.pkl")
        self.assertEqual(data_any["type"], "large")
        
        # Load with specific prefix
        loaded_small = ExpLogger.load_logger("prefix_test", prefix="small", base_path=self.base_path)
        data_small = loaded_small.load_data("data.pkl")
        self.assertEqual(data_small["type"], "small")

    def test_load_logger_nonexistent(self):
        """Test loading nonexistent experiment raises error."""
        with self.assertRaises(FileNotFoundError):
            ExpLogger.load_logger("nonexistent_exp", base_path=self.base_path)
        
        # Test nonexistent prefix
        ExpLogger("test_exp", prefix="existing", base_path=self.base_path)
        with self.assertRaises(FileNotFoundError):
            ExpLogger.load_logger("test_exp", prefix="nonexistent", base_path=self.base_path)

    # Utility and Error Handling Tests
    def test_path_and_figpath(self):
        """Test path utility methods."""
        logger = ExpLogger("test_exp", base_path=self.base_path)
        
        # Test path method
        path = logger.path("test_file.txt")
        expected = logger.experiment_folder() / "test_file.txt"
        self.assertEqual(path, expected)
        
        # Test get_figpath
        figpath = logger.get_figpath("test_figure")
        expected = logger.experiment_folder() / "test_figure.svg"
        self.assertEqual(figpath, expected)
        
        figpath_png = logger.get_figpath("test_figure", "png")
        expected_png = logger.experiment_folder() / "test_figure.png"
        self.assertEqual(figpath_png, expected_png)

    def test_context_manager(self):
        """Test ExpLogger as context manager."""
        with ExpLogger("test_exp", base_path=self.base_path) as logger:
            self.assertIsInstance(logger, ExpLogger)
            logger.save_data(self.test_data, "context_test.pkl")
        
        # Verify data was saved
        logger2 = ExpLogger("test_exp", exp_id=logger.exp_id, base_path=self.base_path)
        loaded_data = logger2.load_data("context_test.pkl")
        self.assertEqual(loaded_data, self.test_data)

    def test_error_handling(self):
        """Test various error conditions."""
        logger = ExpLogger("test_exp", base_path=self.base_path)
        
        # Nonexistent data/model should return None/empty dict
        self.assertIsNone(logger.load_data("nonexistent.pkl"))
        self.assertIsNone(logger.load_model(SimpleModel, "nonexistent.pt"))
        self.assertEqual(logger.load_models(SimpleModel, "nonexistent.pt"), {})

    def test_print_summary(self):
        """Test print_summary method."""
        logger = ExpLogger("test_exp", base_path=self.base_path)
        logger.print_summary()  # Should not crash
        
        # Test with tmp mode and some data
        tmp_logger = ExpLogger("test_exp", tmp=True, base_path=self.base_path)
        tmp_logger.save_data({"test": "data"}, "test.pkl")
        tmp_logger.print_summary()  # Should not crash


if __name__ == "__main__":
    unittest.main(verbosity=2)
