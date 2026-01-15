from __future__ import annotations
from deprecated import deprecated
from pathlib import Path
from datetime import datetime
import pickle as pkl
import traceback
from typing import Any, Dict, List, Optional, Union
import re
import masterthesis.utils as utils
from .icfg import ICfg


class ExpLogger:
    """
    Experiment logger for saving results to disk or memory.

    ## Versioning Conventions:
    - Each experiment should be versioned in the prefix.
    - The prefix should have the format `<run_id_version>`, for example `large_v0-1`, `small_v1-1`, `test_v0-1`.
    - As conventionally, the first number should be the major version, and the second number the minor version.
    - Breaking changes should increment the major version.
    """
    
    def __init__(self, base_name: str, prefix: str = "run", tmp: bool = False, base_path: Optional[Path] = None, exp_id: Optional[str] = None, timestamp: Optional[datetime] = None):
        """Initialize experiment logger.
        
        Args:
            base_name: Base experiment name
            prefix: Run prefix (e.g., "large", "extended")
            tmp: If True, store in a special disk folder <prefix>_tmp; if False, save to disk as usual
            base_path: Custom experiments folder path
            exp_id: Custom experiment ID (for loading existing experiments)
            timestamp: Creation timestamp (for persistent naming)
        """
        self.exp_name = base_name
        self.tmp = tmp
        self.prefix = prefix
        self.base_path = base_path or Path(__file__).parents[2] / "experiments"
        # Always store timestamp
        self.timestamp = timestamp or datetime.now()

        if exp_id is not None:
            self.exp_id = exp_id
        else:
            if self.tmp:
                self.exp_id = f"{prefix}_tmp"
            else:
                timestamp_str = self.timestamp.strftime("%Y_%m_%d_%H_%M_%S")
                self.exp_id = f"{prefix}_{timestamp_str}"
        # Always create the experiment folder (even in tmp mode, now always on disk)
        self.experiment_folder().mkdir(parents=True, exist_ok=True)

    #------------ File and folder management -------------#

    def make_tmp(self):
        """Convert a persistent run to a tmp run: copy to <prefix>_tmp and remove original."""
        if self.tmp:
            print("Already in tmp mode.")
            return
        import shutil
        src = self.experiment_folder()
        tmp_exp_id = f"{self.prefix}_tmp"
        dst = self.base_path / self.exp_name / tmp_exp_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        shutil.rmtree(src)
        self.tmp = True
        self.exp_id = tmp_exp_id
        print(f"Moved experiment to tmp folder: {dst}")

    def make_persistent(self):
        """Convert a tmp run to a persistent run: copy to timestamped folder and remove tmp."""
        if not self.tmp:
            print("Already in persistent mode.")
            return
        import shutil
        timestamp_str = self.timestamp.strftime("%Y_%m_%d_%H_%M_%S")
        new_exp_id = f"{self.prefix}_{timestamp_str}"
        src = self.experiment_folder()
        dst = self.base_path / self.exp_name / new_exp_id
        if dst.exists():
            raise FileExistsError(f"Persistent folder already exists: {dst}")
        shutil.copytree(src, dst)
        shutil.rmtree(src)
        self.tmp = False
        self.exp_id = new_exp_id
        print(f"Moved experiment to persistent folder: {dst}")

    def make_ref(self):
        """
        Copy the current folder to base_path/<exp_name>/<prefix>_ref.
        A reference run is a persistent run that is not modified anymore, and where the results are usually used in a document.
        This does not remove or modify the original folder.
        """
        import shutil
        src = self.experiment_folder()
        ref_exp_id = f"{self.prefix}_ref"
        dst = self.base_path / self.exp_name / ref_exp_id
        if dst.exists():
            print(f"Reference folder already exists: {dst}, overwriting...")
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied experiment to reference folder: {dst}")

    def experiment_folder(self) -> Path:
        """Return experiment folder path."""
        return self.base_path / self.exp_name / self.exp_id
    
    def remove_empty_folders(self) -> None:
        """
        Remove empty folders in the experiment path.
        These are usually from interrupted or errored runs.
        """
        try:
            exp_base_folder = self.base_path / self.exp_name
            for folder in exp_base_folder.glob("*"):
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
        except Exception as e:
            print(f"Error removing empty folders: {e}")
            traceback.print_exc()

    def remove_old_runs(self, keep: int = 5, cur_prefix: bool = True) -> None:
        """
        Remove old runs, keeping the most recent `keep` runs.
        Args:
            keep: Number of runs to keep
            cur_prefix: If True, only consider runs with the current prefix. Otherwise, consider all runs.
        """
        self.remove_empty_folders()
        try:
            exp_base_folder = self.base_path / self.exp_name
            
            folders = [d for d in exp_base_folder.iterdir() if d.is_dir()]
            
            if cur_prefix:
                folders = [d for d in folders if d.name.startswith(f"{self.prefix}_")]
            
            def extract_timestamp(folder):
                match = re.search(r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$", folder.name)
                return datetime.strptime(match.group(1), "%Y_%m_%d_%H_%M_%S") if match else datetime.max
            
            folders.sort(key=extract_timestamp, reverse=True)
            
            for folder in folders[keep:]:
                import shutil
                shutil.rmtree(folder)
                
        except Exception as e:
            print(f"Error removing old runs: {e}")
            traceback.print_exc()
        

        
    
    # ----------- Logging and storing data ---------------#

    def log_cfg(self, cfg: ICfg, filename: str = "cfg.txt") -> None:
        """Log configuration to file."""
        filepath = self.experiment_folder() / filename
        try:
            with open(filepath, 'w') as f:
                cfg.print_config(file = f)
        except Exception as e:
            print(f"Error saving configuration to {filename}: {e}")
            traceback.print_exc()

    def save_figure(self, figure: Any, figname: str, force_save = False, show = False, pickle = False) -> None:
        """Save figure to disk. Call before showing/closing figure. If `pickle` is True, also save to pickle file."""
        self._save_figure(figure, figname)
        if pickle:
            try:
                with open(self.experiment_folder() / f"{figname}.pkl", 'wb') as f:
                    pkl.dump(figure, f)
            except Exception as e:
                print(f"Error saving figure {figname} to pickle: {e}")
                traceback.print_exc()

        if show:
            try:
                import IPython.display
                import matplotlib.pyplot as plt

                IPython.display.display(figure)
                # Close the figure to prevent double display
                plt.close(figure)
            except Exception as e:
                print(f"Error showing figure {figname}: {e}")
                traceback.print_exc()

    def path(self, fname: str) -> Path:
        """Get full path for file in experiment folder."""
        return self.experiment_folder() / fname

    def _save_figure(self, figure: Any, figname: str) -> None:
        """Save figure with error handling."""
        def save_plotly():
            figure.write_html(self.experiment_folder() / f"{figname}.html")
            
        def save_matplotlib():
            figure.savefig(self.experiment_folder() / f"{figname}.svg", format='svg')
        
        if hasattr(figure, 'write_image'):
            self._tryit(save_plotly, f"saving plotly figure {figname}")
        elif hasattr(figure, 'savefig'):
            self._tryit(save_matplotlib, f"saving matplotlib figure {figname}")
        else:
            print(f"Warning: Unknown figure type for {figname}, cannot save")

    def save_data(self, data: Any, filename: str) -> None:
        """Save data to pickle file."""
        filepath = self.experiment_folder() / filename
        self._tryit(lambda: self._save_pickle(data, filepath), f"saving data to {filename}")

    def load_data(self, filename: str = "data.pkl") -> Any:
        """Load data from pickle file."""
        filepath = self.experiment_folder() / filename
        try:
            return self._load_pickle(filepath)
        except Exception as e:
            print(f"Error loading data from {filename}: {e}")
            return None

    def print_summary(self) -> None:
        """Print experiment summary."""
        print(f"Experiment: {self.exp_name}")
        print(f"ID: {self.exp_id}")
        print(f"Folder: {self.experiment_folder()}")
        print(f"Temporary mode: {self.tmp}")

    def get_figpath(self, figname: str, extension: str = "svg") -> Path:
        """Get figure file path."""
        return self.experiment_folder() / f"{figname}.{extension}"

    @staticmethod
    def load_logger(exp_name: str, prefix: Optional[str] = None, base_path: Optional[Path] = None, load_tmp: bool = False) -> ExpLogger:
        """Load latest experiment logger for given experiment name and optional prefix.
        Args:
            exp_name: The experiment name to load
            prefix: Optional prefix to filter by. If None, searches all prefixes
            base_path: Custom base path for experiments folder
            load_tmp: If True, load the <prefix>_tmp run instead of a timestamped run
        Returns:
            ExpLogger instance for the latest matching run
        """
        if base_path is None:
            base_path = Path(__file__).parents[2] / "experiments"

        exp_path = base_path / exp_name
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment folder not found for {exp_name} at {exp_path}")

        if load_tmp:
            # Load the tmp run
            if prefix is None:
                prefix = "run"
            exp_id = f"{prefix}_tmp"
            folder = exp_path / exp_id
            if not folder.exists():
                raise FileNotFoundError(f"No tmp run found for experiment {exp_name} with prefix '{prefix}' at {exp_path}")
            # Try to get timestamp from files if possible, else use now
            timestamp = datetime.now()
            return ExpLogger(base_name=exp_name, exp_id=exp_id, base_path=base_path, prefix=prefix, tmp=True, timestamp=timestamp)

        # Find all run folders for the experiment
        run_folders = [d for d in exp_path.iterdir() if d.is_dir()]

        if not run_folders:
            raise FileNotFoundError(f"No runs found for experiment {exp_name} at {exp_path}")

        # Filter by prefix if specified
        if prefix is not None:
            run_folders = [d for d in run_folders if d.name.startswith(f"{prefix}_")]
            if not run_folders:
                raise FileNotFoundError(f"No runs found for experiment {exp_name} with prefix '{prefix}' at {exp_path}")

        def extract_timestamp(folder):
            match = re.search(r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$", folder.name)
            if match:
                return datetime.strptime(match.group(1), "%Y_%m_%d_%H_%M_%S")
            else:
                return datetime.min
        
        latest_run_folder = max(run_folders, key=extract_timestamp)
        exp_id = latest_run_folder.name
        timestamp = extract_timestamp(latest_run_folder)
        return ExpLogger(base_name=exp_name, exp_id=exp_id, base_path=base_path, prefix=prefix, tmp=False, timestamp=timestamp)

    @staticmethod
    def _save_pickle(data: Any, filepath: Path) -> None:
        """Save data to pickle file."""
        with open(filepath, 'wb') as f:
            pkl.dump(data, f)

    @staticmethod
    def _load_pickle(filepath: Path) -> Any:
        """Load data from pickle file."""
        with open(filepath, 'rb') as f:
            return pkl.load(f)

    @staticmethod
    def _tryit(func, msg: str) -> None:
        """Try function and print error with traceback if it fails."""
        try:
            func()
        except Exception as e:
            print(f"Error: {msg}")
            print(e)
            traceback.print_exc()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @deprecated("Use save_models_cfgs_new instead")
    def save_model(self, model, filename: str = "model.pt") -> None:
        """Save PyTorch model to disk."""
        import torch
        try:
            filepath = self.experiment_folder() / filename
            torch.save(model.state_dict(), filepath)
        except Exception as e:
            print(f"Error saving model to {filename}: {e}")

    @deprecated("Use load_models_cfgs_new instead")
    def load_model(self, model_class, filename: str = "model.pt", map_location=None):
        """Load PyTorch model from disk."""
        import torch
        try:
            model = model_class()
            filepath = self.experiment_folder() / filename
            state_dict = torch.load(filepath, map_location=map_location)
            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            print(f"Error loading model from {filename}: {e}")
            return None

    @deprecated("Use save_models_cfgs_new instead")
    def save_models(self, models: dict, filename: str = "models.pt") -> None:
        """Save multiple PyTorch models to disk."""
        import torch
        try:
            state_dicts = {name: model.state_dict() for name, model in models.items()}
            filepath = self.experiment_folder() / filename
            torch.save(state_dicts, filepath)
        except Exception as e:
            print(f"Error saving models to {filename}: {e}")

    @deprecated("Use load_models_cfgs_new instead")
    def load_models(self, model_classes, filename: str = "models.pt", map_location=None) -> dict:
        """Load multiple PyTorch models from disk."""
        import torch
        try:
            filepath = self.experiment_folder() / filename
            state_dicts = torch.load(filepath, map_location=map_location)
            if isinstance(model_classes, dict):
                class_dict = model_classes
            else:
                class_dict = {name: model_classes for name in state_dicts.keys()}
            models = {}
            for name, model_class in class_dict.items():
                model = model_class()
                if name in state_dicts:
                    model.load_state_dict(state_dicts[name])
                else:
                    print(f"Warning: No state dict found for model '{name}' in {filename}")
                models[name] = model
            return models
        except Exception as e:
            print(f"Error loading models from {filename}: {e}")
            return {}
        

    def save_models_cfgs_new(self, models_dict : Dict, cfgs_dict : Dict, prefix = "") -> None:
        """Save multiple models and their configurations to disk.
        Cfgs should be a dictionary mapping model keys to their respective ICfg.
        The ICfg should have a model attribute that gives the subconfiguration for the model class.
        """
        assert models_dict.keys() == cfgs_dict.keys(), "Models and configs must have the same keys"
        import torch
        try:
            state_dicts = {name: model.state_dict() for name, model in models_dict.items()}
            filepath = self.experiment_folder() / (prefix + "models.pt")
            torch.save(state_dicts, filepath)
            cfg_filepath = self.experiment_folder() / (prefix + "cfgs.pkl")
            with open(cfg_filepath, 'wb') as f:
                pkl.dump(cfgs_dict, f)
        except Exception as e:
            print(f"Error saving models and configs: {e}")

    def load_models_cfgs_new(self, prefix: str = "", map_location=None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Load models using saved ICfg configurations."""
        import torch
        try:
            map_location = map_location or utils.get_free_gpu()
            state_dicts = torch.load(self.experiment_folder() / (prefix + "models.pt"), map_location=map_location)
            with open(self.experiment_folder() / (prefix + "cfgs.pkl"), 'rb') as f:
                cfgs_saved = pkl.load(f)
            models = {}
            for name in state_dicts.keys():
                # Get saved config
                model_cfg = cfgs_saved.get(name)
                if not model_cfg:
                    print(f"No saved config found for model '{name}'")
                    continue
                # Instantiate and load
                try:
                    model = model_cfg.instantiate()
                    model.load_state_dict(state_dicts[name])
                    models[name] = model
                except Exception as e:
                    print(f"Failed loading model '{name}': {e}")
            return models, cfgs_saved
        except Exception as e:
            print(f"Error loading models: {e}")
            return {}, {}
