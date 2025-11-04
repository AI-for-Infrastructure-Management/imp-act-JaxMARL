import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union


class Run:
    """
    Class representing a single run from a WandB experiment that was downloaded
    using wandb_utils.py.
    """
    
    def __init__(self, run_path: Union[str, Path]):
        """
        Initialize a Run object by loading data from the given path.
        
        Args:
            run_path: Path to the directory containing run data (config.yaml, summary.yaml, history.csv)
        """
        self.path = Path(run_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Run path {self.path} does not exist.")

        self.id = self.path.name
        
        # Load config
        config_path = self.path / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file {config_path} does not exist.")
            
        # Load summary
        summary_path = self.path / 'summary.yaml'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.summary = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Summary file {summary_path} does not exist.")

        self.id = self.summary.get('id', self.id)
        self.name = self.summary.get('name', None) 
        self.url = self.summary.get('url', None) 
        self.state = self.summary.get('state', "unknown") 

        # Load history
        history_path = self.path / 'history.csv'
        if history_path.exists():
            self.history = pd.read_csv(history_path)
        else:
            self.history = None

    
    def get_history_values(self, key: str) -> pd.Series:
        """
        Get values for a specific metric from the run history.
        
        Args:
            key: Name of the metric to extract from history
            
        Returns:
            Series containing the values for the specified metric
        """
        if key in self.history:
            return self.history[key]
        return None
    
    def __str__(self) -> str:
        return f"Run {self.id} ({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class Sweep:
    """
    Class representing a sweep from a WandB project that was downloaded
    using wandb_utils.py.
    """
    
    def __init__(self, sweep_path: Union[str, Path]):
        """
        Initialize a Sweep object by loading data from the given path.
        
        Args:
            sweep_path: Path to the directory containing sweep data
        """
        self.path = Path(sweep_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Run path {self.path} does not exist.")
        
        self.name = self.path.name
        
        # Load sweep summary
        summary_path = self.path / 'sweep_summary.yaml'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary_data = yaml.safe_load(f)
                self.id = summary_data.get('id', '')
                self.config = summary_data.get('config', {})
                self.entity = summary_data.get('entity', '')
                self.project = summary_data.get('project', '')
                self.url = summary_data.get('url', '')
                self.run_ids = summary_data.get('runs', {})
                self.name = summary_data.get('name', self.name)
        else:
            self.id = ''
            self.config = {}
            self.entity = ''
            self.project = ''
            self.url = ''
            self.run_ids = {}
        
        # Lazy loading for runs
        self._runs = None
    
    @property
    def runs(self) -> Dict[str, Run]:
        """
        Lazily load all runs in this sweep
        
        Returns:
            Dictionary of run ID to Run object
        """
        if self._runs is None:
            self._runs = {}
            # Load each run
            for run_id in self.run_ids:
                run_path = self.path / run_id
                if run_path.exists():
                    self._runs[run_id] = Run(run_path)
        
        return self._runs
    
    def get_run(self, run_id: str) -> Optional[Run]:
        """
        Get a specific run by ID
        
        Args:
            run_id: ID of the run to retrieve
            
        Returns:
            Run object if found, None otherwise
        """
        return self.runs.get(run_id)
    
    def __str__(self) -> str:
        return f"Sweep {self.id} ({self.name}) with {len(self.run_ids)} runs"
    
    def __repr__(self) -> str:
        return self.__str__()


class Project:
    """
    Class representing a WandB project that was downloaded
    using wandb_utils.py..
    """
    
    def __init__(self, project_path: Union[str, Path]):
        """
        Initialize a Project object by loading data from the given path.
        
        Args:
            project_path: Path to the directory containing project data
        """
        self.path = Path(project_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Project path {self.path} does not exist.")
        
        self.name = self.path.name
        
        # Load project summary
        summary_path = self.path / 'project_summary.yaml'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary_data = yaml.safe_load(f)
                self.entity = summary_data.get('entity', '')
                self.url = summary_data.get('url', '')
                self.sweep_info = summary_data.get('sweeps', {})
        else:
            raise FileNotFoundError(f"Project summary file {summary_path} does not exist.")
        
        # Lazy loading for sweeps
        self._sweeps = None
    
    @property
    def sweeps(self) -> Dict[str, Sweep]:
        """
        Lazily load all sweeps in this project
        
        Returns:
            Dictionary of sweep ID to Sweep object
        """
        if self._sweeps is None:
            self._sweeps = {}
            # Load each sweep
            for sweep_id, sweep_data in self.sweep_info.items():
                sweep_path = self.path / sweep_data.get('name', '')
                if sweep_path.exists():
                    self._sweeps[sweep_id] = Sweep(sweep_path)
        
        return self._sweeps
    
    def get_sweep(self, sweep_id: str) -> Optional[Sweep]:
        """
        Get a specific sweep by ID
        
        Args:
            sweep_id: ID of the sweep to retrieve
            
        Returns:
            Sweep object if found, None otherwise
        """
        return self.sweeps.get(sweep_id)
    
    def get_sweep_by_name(self, name: str) -> Optional[Sweep]:
        """
        Get a specific sweep by name
        
        Args:
            name: Name of the sweep to retrieve
            
        Returns:
            Sweep object if found, None otherwise
        """
        for sweep in self.sweeps.values():
            if sweep.name == name:
                return sweep
        return None
    
    def __str__(self) -> str:
        return f"Project {self.name} with {len(self.sweep_info)} sweeps"
    
    def __repr__(self) -> str:
        return self.__str__()


# Helper functions for loading data
def load_project(project_path: Union[str, Path]) -> Project:
    """
    Load a WandB project from the specified path
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Project object containing the loaded data
    """
    return Project(project_path)


def load_sweep(sweep_path: Union[str, Path]) -> Sweep:
    """
    Load a WandB sweep from the specified path
    
    Args:
        sweep_path: Path to the sweep directory
        
    Returns:
        Sweep object containing the loaded data
    """
    return Sweep(sweep_path)


def load_run(run_path: Union[str, Path]) -> Run:
    """
    Load a WandB run from the specified path
    
    Args:
        run_path: Path to the run directory
        
    Returns:
        Run object containing the loaded data
    """
    return Run(run_path)


# Example usage
if __name__ == "__main__":
    # Load a project
    project = load_project("evaluation/data/jaxMARL-CologneBonnDusseldorf-v1")
    print(f"Loaded project: {project}")
    
    # Load a sweep
    if project.sweeps:
        sweep_id = next(iter(project.sweeps))
        sweep = project.sweeps[sweep_id]
        print(f"Loaded sweep: {sweep}")
        
        # Load a run
        if sweep.runs:
            run_id = next(iter(sweep.runs))
            run = sweep.runs[run_id]
            print(f"Loaded run: {run}")
            
            # Show some metrics
            if not run.history.empty:
                print(f"History columns: {run.history.columns.tolist()}")
                if 'test_returned_episode_returns' in run.history:
                    values = run.history['test_returned_episode_returns']
                    print(f"Best test return: {values.max()}")