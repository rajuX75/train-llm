import yaml
from dataclasses import fields, is_dataclass
from typing import Any, Type, TypeVar

T = TypeVar('T')

class ConfigLoader:
    """A flexible loader for hierarchical configurations."""

    def __init__(self, config_data: dict[str, Any]):
        """
        Initializes the loader with configuration data.

        Args:
            config_data: A dictionary containing configuration settings.
        """
        self._config = config_data

    @classmethod
    def from_yaml(cls, file_path: str) -> 'ConfigLoader':
        """
        Loads configuration from a YAML file.

        Args:
            file_path: The path to the YAML configuration file.

        Returns:
            A new ConfigLoader instance.
        """
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(config_data)

    def get_section(self, section_name: str, config_class: Type[T]) -> T:
        """
        Loads a specific section of the configuration into a dataclass.

        Args:
            section_name: The name of the configuration section (e.g., 'model').
            config_class: The dataclass to populate with the section's data.

        Returns:
            An instance of the provided dataclass with loaded values.

        Raises:
            ValueError: If the section is not found or if there are unknown keys.
        """
        section_data = self._config.get(section_name)
        if section_data is None:
            raise ValueError(f"Configuration section '{section_name}' not found.")

        if not is_dataclass(config_class):
            raise TypeError("config_class must be a dataclass.")

        # Get valid field names from the dataclass
        valid_keys = {f.name for f in fields(config_class)}

        # Filter out unknown keys from the loaded data
        filtered_data = {k: v for k, v in section_data.items() if k in valid_keys}

        # Check for unexpected keys
        unknown_keys = set(section_data.keys()) - valid_keys
        if unknown_keys:
            raise ValueError(f"Unknown configuration keys in section '{section_name}': {unknown_keys}")

        return config_class(**filtered_data)