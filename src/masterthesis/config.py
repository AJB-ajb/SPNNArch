"""
Configuration utilities for the MasterThesis project.

This module provides a flexible configuration class that supports both permanent
and temporary attributes, with dictionary-like behavior and serialization support.
"""

from collections.abc import MutableMapping
import dataclasses


class Cfg(MutableMapping):
    """
    A configuration class that behaves like a dictionary but allows for temporary keys.

    This class is useful for managing configurations in experiments, allowing for both
    permanent and temporary attributes. Temporary attributes are marked via the set
    `_tmp_keys`. Temporary attributes are not serialized and are meant for intermediate
    results or non-serializable data.

    The class supports dictionary-like operations and provides methods for updating,
    copying, and printing summaries of the configuration.

    Note: This class has two distinct ways to set attributes to avoid name issues:
        cfg['key'] = value   # allows adding new keys
        cfg.key = value      # doesn't allow adding new keys, only updating existing ones
    """
    def __init__(self, dct=None, **kwargs):
        # Initialize _tmp_keys first to avoid AttributeError in __setattr__
        object.__setattr__(self, '_tmp_keys', set())
        
        if dct is not None:
            self.__dict__.update(dct)
        self.__dict__.update(kwargs)

    def __getstate__(self):
        """
        Returns the state of the config object for serialization.
        Excludes non-serializable (temporary) attributes.
        """
        state = self.__dict__.copy()
        # Remove temporary keys from serialization
        for key in self._tmp_keys:
            state.pop(key, None)
        # Don't serialize the _tmp_keys set itself
        state.pop('_tmp_keys', None)
        return state
    
    def __setstate__(self, state):
        """Restore state and reinitialize _tmp_keys"""
        self.__dict__.update(state)
        self._tmp_keys = set()
    
    def set_tmp(self, key, value):
        """
        Set a temporary value in the config object.
        This is useful for storing non-serializable attributes or intermediate results.
        """
        self.__dict__[key] = value
        self._tmp_keys.add(key)

    def update_tmp(self, **kwargs):
        """Update multiple temporary values at once"""
        for key, value in kwargs.items():
            self.set_tmp(key, value)
    
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def __delitem__(self, key):
        del self.__dict__[key]
        self._tmp_keys.discard(key)  # Remove from tmp_keys if it was temporary
        
    def __iter__(self):
        return iter(self.__dict__)
        
    def __len__(self):
        return len(self.__dict__)

    def __setattr__(self, name, value):
        # Allow setting _tmp_keys during initialization
        if name == '_tmp_keys':
            object.__setattr__(self, name, value)
        elif hasattr(self, '__dict__') and name in self.__dict__:
            self[name] = value
        else:
            # Allow new attributes to be set normally
            self.__dict__[name] = value

    def keys(self):
        return self.__dict__.keys()
    
    def values(self):
        return self.__dict__.values()
    
    def items(self):
        return self.__dict__.items()
        
    def copy(self):
        """Create a copy including temporary keys tracking"""
        new_cfg = Cfg(self.__dict__)
        new_cfg._tmp_keys = self._tmp_keys.copy()
        return new_cfg
        
    def __repr__(self):
        return f'Cfg({self.__dict__.__str__()})'
    
    def str_id(self):
        """
        Returns a string representation of the config, suitable for use as an identifier, 
        based on the hash of non-temporary config items only.
        """
        # Only hash permanent (non-temporary) items for stable IDs
        permanent_items = {k: v for k, v in self.__dict__.items() 
                          if k not in self._tmp_keys and k != '_tmp_keys'}
        return f"cfg_{hash(frozenset(permanent_items.items())):x}"
        
    def replace(self, tmp_kwargs=None, **kwargs):
        """Create a new config with updated values"""
        new_cfg = self.copy()
        new_cfg.update(kwargs)
        if tmp_kwargs:
            new_cfg.update_tmp(**tmp_kwargs)
        return new_cfg
    
    def print_summary(self):
        """
        Print a summary of the config, showing all keys and their values.
        Separates permanent and temporary attributes.
        """
        if 'name' in self.keys():
            print(f"Config {self['name']} Summary:")
        else:
            print("Config Summary:")
        print("-" * 20)
        
        # Print permanent attributes
        print("Permanent attributes:")
        for key, value in self.items():
            if key not in self._tmp_keys and key != '_tmp_keys':
                print(f"  {key}: {value}")
        
        # Print temporary attributes if any
        if self._tmp_keys:
            print("Temporary attributes:")
            for key in self._tmp_keys:
                if key in self.__dict__:
                    print(f"  {key}: {self.__dict__[key]}")
        print("-" * 20)
    
    def validate_required(self, required_keys):
        """
        Validate that all required keys are present in the configuration.
        
        Args:
            required_keys: List of required key names
            
        Raises:
            ValueError: If any required keys are missing
        """
        missing_keys = [key for key in required_keys if key not in self.__dict__]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    def get_summary(self):
        """
        Get a summary of the configuration for debugging.
        
        Returns:
            Dict with configuration summary
        """
        permanent_keys = {k: v for k, v in self.__dict__.items() 
                         if k not in self._tmp_keys and k != '_tmp_keys'}
        temp_keys = {k: v for k, v in self.__dict__.items() 
                    if k in self._tmp_keys}
        
        return {
            'permanent_keys': list(permanent_keys.keys()),
            'temporary_keys': list(temp_keys.keys()),
            'total_keys': len(self.__dict__) - 1,  # -1 for _tmp_keys
        }


def replace(d, **kwargs):
    """
    Create a copy of a configuration object with updated values.
    
    Supports Cfg objects, dataclasses, and regular dictionaries.
    
    Args:
        d: The object to copy and update (Cfg, dataclass, or dict)
        **kwargs: Key-value pairs to update in the copy
        
    Returns:
        A new object of the same type with updated values
    """
    if type(d) == Cfg:
        return d.replace(**kwargs)
    elif hasattr(d, "__dataclass_fields__"):  # Check if it's a dataclass
        return dataclasses.replace(d, **kwargs)
    else:
        d_copy = d.copy()
        d_copy.update(kwargs)
        return d_copy
