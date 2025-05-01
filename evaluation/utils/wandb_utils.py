import numpy as np
import argparse
import wandb
import pandas
import yaml
from pathlib import Path
import re

api = wandb.Api()

# Define the keys to download from the run history.
# If the history contains any of these keys, the data will be downloaded.
HISTORY_KEYS_TO_DOWNLOAD = [
        'env_step',
        'returned_episode_returns',
        '_runtime',
        'rnorm_std',
        'update_steps',
        '_timestamp',
        'rnorm_mean',
        'test_returned_episode',
        'grad_steps',
        'qvals',
        'loss',
        '_step',
        'test_returned_episode_returns'
    ]

def download_run(run: wandb.apis.public.Run | str) -> dict:
    """
    Downloads data for a single W&B run.
    
    Args:
        run: The W&B Run object to download data for.
        
    Returns:
        dict: A dictionary containing the run's data, including name, ID, 
              history, config, summary, and URL.
    """
    if isinstance(run, str):
        run = api.run(run)

    match_keys = [k for k in run.history().keys() if k in HISTORY_KEYS_TO_DOWNLOAD]

    history = pandas.DataFrame([d for d in run.scan_history(keys=match_keys)])

    run_data = {
        'name': run.name,
        'id': run.id,
        'history': history,
        'config': dict(run.config),
        'summary': {k:v for k,v in run.summary.items() if k not in ['_wandb', 'gpu_stats']},
        'url': run.url,
    }

    run_data['summary']['id'] = run_data['id']
    run_data['summary']['name'] = run_data['name']
    run_data['summary']['url'] = run_data['url']
    run_data['summary']['state'] = run.state
    
    return run_data

def store_run_data(run_data: dict, path: Path) -> None:
    """
    Stores the downloaded run data to the filesystem.
    
    Args:
        run_data: Dictionary containing the run data to store.
        path: Path where the run data will be stored.
    """
    run_path = Path.joinpath(path, run_data['id'])
    run_path.mkdir(parents=True, exist_ok=True)

    # save the config
    with open(f'{run_path}/config.yaml', 'w') as f:
        yaml.dump(run_data['config'], f)

    # save the summary
    with open(f'{run_path}/summary.yaml', 'w') as f:
        yaml.dump(run_data['summary'], f)

    # save the history
    run_data['history'].to_csv(f'{run_path}/history.csv')

def download_sweep(sweep_id: str, path: str) -> dict:
    """
    Downloads data for a W&B sweep and all its runs.
    
    Args:
        sweep_id: The ID of the W&B sweep to download in the format 'entity/project/sweep_id'.
        path: Directory path where the sweep data will be stored.
        
    Returns:
        dict: A high-level summary of the sweep, including ID, name, runs, and URL.
    """
    sweep = api.sweep(sweep_id)
    print(f'Downloading data for sweep {sweep.name}')

    sweep_path = Path.joinpath(Path(path), sweep.name)
    sweep_path.mkdir(parents=True, exist_ok=True)

    sweep_data = {
        'entity': sweep.entity,
        'project': sweep.project,
        'name': sweep.name,
        'id': sweep.id,
        'config': dict(sweep.config),
        'url': sweep.url,
        'runs': {}
    }

    for run in sweep.runs:
        sweep_data['runs'][run.id] = run.name
        if run.state != 'finished':
            print(f'Run {run.name} is not finished, skipping')
            continue
        if Path.joinpath(sweep_path, run.id).exists():
            print(f'Run {run.name} already exists, skipping')
            continue
        
        print(f'Downloading data for run {run.name}')

        run_data = download_run(run)
        store_run_data(run_data, sweep_path)


    with open(f'{sweep_path}/sweep_summary.yaml', 'w') as f:
        yaml.dump(sweep_data, f)
    
    high_level_summary = {
        k: v for k, v in sweep_data.items() if k in ['id', 'name', 'runs', 'url']
    }

    return high_level_summary

def download_project(entity: str, project: str, path: str) -> None:
    """
    Downloads data for an entire W&B project, including all sweeps and runs.
    
    Args:
        entity: The W&B entity (user or organization) name.
        project: The W&B project name.
        path: Directory path where the project data will be stored.
    """
    project_object = api.project(entity=entity, name=project)
    print(f'Downloading data for project {project_object.name}')
    project_path = Path.joinpath(Path(path), project_object.name)
    project_path.mkdir(parents=True, exist_ok=True)
    project_data = {
        'entity': project_object.entity,
        'name': project_object.name,
        'url': project_object.url,
        'sweeps': {}
    }

    for sweep in project_object.sweeps():
        path = Path.joinpath(project_path, sweep.name)

        sweep_summary = download_sweep('/'.join([entity, project, sweep.id]), project_path)
        sweep_summary['path'] = str(path)
        project_data['sweeps'][sweep.id] = sweep_summary

    with open(f'{project_path}/project_summary.yaml', 'w') as f:
        yaml.dump(project_data, f)

def get_run_from_link(url: str) -> str | None:
    """
    Extracts the 'entity/project/run_id' strings from a W&B run URL.
    
    Args:
        url: The W&B run URL string.
        
    Returns:
        The extracted 'entity', 'project', 'run_id' strings if the URL matches
        the expected format, otherwise None.
    """
    # Regex explanation:
    # (?:https?://)?    - Optional 'http://' or 'https://' (non-capturing group)
    # (?:www\.)?        - Optional 'www.' (non-capturing group)
    # wandb\.ai/        - Literal 'wandb.ai/' (dot is escaped)
    # ([^/]+)           - Capture group 1: Entity name (one or more chars not '/')
    # /                 - Literal '/'
    # ([^/]+)           - Capture group 2: Project name (one or more chars not '/')
    # /runs/            - Literal '/runs/'
    # ([^/?#]+)         - Capture group 3: Run ID (one or more chars not '/', '?', or '#')
    # /?                - Optional trailing slash
    # (?:[?#].*)?       - Optional query string or fragment identifier (non-capturing group)
    regex = r"(?:https?://)?(?:www\.)?wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)/?(?:[?#].*)?"
    
    match = re.search(regex, url)

    if match:
        entity = match.group(1)
        project = match.group(2)
        run_id = match.group(3)
        return entity, project, run_id
    else:
        # Return None if the URL doesn't match the pattern
        return None

def get_sweep_from_link(url: str) -> str | None:
    """
    Extracts the 'entity/project/sweep_id' strings from a W&B sweep URL.

    Args:
        url: The W&B sweep URL string.

    Returns:
        The extracted 'entity', 'project', 'sweep_id' strings if the URL matches
        the expected format, otherwise None.
    """
    # Regex explanation:
    # (?:https?://)?    - Optional 'http://' or 'https://' (non-capturing group)
    # (?:www\.)?        - Optional 'www.' (non-capturing group)
    # wandb\.ai/        - Literal 'wandb.ai/' (dot is escaped)
    # ([^/]+)           - Capture group 1: Entity name (one or more chars not '/')
    # /                 - Literal '/'
    # ([^/]+)           - Capture group 2: Project name (one or more chars not '/')
    # /sweeps/          - Literal '/sweeps/'
    # ([^/?#]+)         - Capture group 3: Sweep ID (one or more chars not '/', '?', or '#')
    # /?                - Optional trailing slash
    # (?:[?#].*)?       - Optional query string or fragment identifier (non-capturing group)
    regex = r"(?:https?://)?(?:www\.)?wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/?#]+)/?(?:[?#].*)?"

    match = re.search(regex, url)

    if match:
        entity = match.group(1)
        project = match.group(2)
        sweep_id = match.group(3)
        return entity, project, sweep_id
    else:
        # Return None if the URL doesn't match the pattern
        return None

def get_project_from_link(url: str) -> str | None:
    """
    Extracts the 'entity/project' path from a W&B project URL.
    Args:
        url: The W&B project URL string.
    Returns:
        The extracted 'entity/project' string if the URL matches
        the expected format, otherwise None.
    """
    # Regex explanation:
    # (?:https?://)?    - Optional 'http://' or 'https://' (non-capturing group)
    # (?:www\.)?        - Optional 'www.' (non-capturing group)
    # wandb\.ai/        - Literal 'wandb.ai/' (dot is escaped)
    # ([^/]+)           - Capture group 1: Entity name (one or more chars not '/')
    # /                 - Literal '/'
    # ([^/]+)           - Capture group 2: Project name (one or more chars not '/')
    # /?                - Optional trailing slash
    # (?:[?#].*)?       - Optional query string or fragment identifier (non-capturing group)
    regex = r"(?:https?://)?(?:www\.)?wandb\.ai/([^/]+)/([^/]+)/?(?:[?#].*)?"
    match = re.search(regex, url)
    if match:
        entity = match.group(1)
        project = match.group(2)
        return entity, project
    else:
        # Return None if the URL doesn't match the pattern
        return None
    


def main(args: dict) -> None:
    """
    Main function to handle downloading W&B data based on command line arguments.
    
    Args:
        args: Dictionary containing command line arguments.
    
    Raises:
        ValueError: If neither project ID/link nor sweep ID/link is provided.
    """
    # Download project data
    if  args['project_id'] is not None:
        project_id = args['project_id']
        entity, project = project_id.split('/')
        download_project(entity, project, args['output_dir'])
    elif args['project_link'] is not None:
        entity, project = get_project_from_link(args['project_link'])
        if entity is None or project is None:
            raise ValueError('Invalid project link provided. Please provide a valid W&B project link.')
        else:
            print(f'Extracted project ID: {entity}/{project}')
        download_project(entity, project, args['output_dir'])
    
    # Download sweep data
    elif args['sweep_id'] is not None:
        sweep_id = args['sweep_id']
        download_sweep(sweep_id, args['output_dir']) 

    elif args['sweep_link'] is not None:
        sweep_id = '/'.join(get_sweep_from_link(args['sweep_link']))
        if sweep_id is None:
            raise ValueError('Invalid sweep link provided. Please provide a valid W&B sweep link.')
        else:
            print(f'Extracted sweep ID: {sweep_id}')
            download_sweep(sweep_id, args['output_dir']) 
    else:
        raise ValueError('Please provide either a sweep ID or a sweep link')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', dest='project_id', type=str, default=None, help='WandB project ID (entity/project)')
    parser.add_argument('--project-link', dest='project_link', type=str, default=None, help='WandB project link (https://wandb.ai/entity/project)')
    parser.add_argument('--sweep-id', dest='sweep_id', type=str, default=None, help='WandB sweep ID (entity/project/sweep_id)')
    parser.add_argument('--sweep-link', dest='sweep_link', type=str, default=None, help='WandB sweep link (https://wandb.ai/entity/project/sweeps/sweep_id)')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='data', help='Directory to save the downloaded data')

    try: 
        main(vars(parser.parse_args()))
    except ValueError as e:
        print(f"An error occurred: {e}")
        parser.print_help()
        exit(1)