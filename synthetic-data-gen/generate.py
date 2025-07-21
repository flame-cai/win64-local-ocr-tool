# /generate.py

import yaml
from pathlib import Path
import argparse

from manuscript_generator.configs.base_config import Config
from manuscript_generator.generator import SyntheticManuscriptGenerator
# Import these to ensure they are registered
from manuscript_generator.layout_strategies import *
from manuscript_generator.augmentations import *

def main():
    """
    Main execution function.
    This is not a CLI, but a simple script to drive the generator.
    To change parameters, edit the YAML file.
    """
    # Use argparse for a simple --dry-run flag, as requested.
    parser = argparse.ArgumentParser(
        description="""
        Python-based synthetic manuscript data generator.
        Configuration is managed via the YAML file.
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate a small number of samples to test the configuration.'
    )
    args = parser.parse_args()

    # 1. Load and validate configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return

    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    try:
        config = Config(**config_dict)
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return

    # 2. Initialize the generator
    generator = SyntheticManuscriptGenerator(config)

    # 3. Run the generation process
    generator.generate_dataset(dry_run=args.dry_run)

if __name__ == "__main__":
    main()