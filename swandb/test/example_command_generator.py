"""
Example command generator module.

This can be referenced from the CLI using:
path/to/example_command_generator.py::echo
"""
import pandas as pd
from typing import Dict, Any, List

def echo(
    runs_df: pd.DataFrame,
    array_mapping: Dict[int, Any], 
    grouping_key: str
) -> Dict[int, List[str]]:
    """
    Example command generator that creates a Python command for each group.
    
    Args:
        runs_df: The full merged DataFrame
        array_mapping: Dict mapping array indices to group IDs
        grouping_key: The column used for grouping
        
    Returns:
        Dict mapping array indices to lists of command arguments
    """
    commands = {}
    for array_idx, group_id in array_mapping.items():
        # Get rows for this group
        group_df = runs_df[runs_df[grouping_key] == group_id]
        
        # Get run IDs as a comma-separated string
        run_ids = ",".join(str(run_id) for run_id in group_df["run_id"].tolist())
        
        commands[array_idx] = ["echo", "sleeping 3", ";", "sleep", "3", ";", "echo", array_idx, run_ids]
    return commands