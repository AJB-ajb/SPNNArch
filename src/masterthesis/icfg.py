"""
Instantiable Configuration (ICfg) system for experiment configuration.

This module provides a minimal, explicit configuration system that allows
defining experiment configurations that can be instantiated into actual objects.
"""

from collections.abc import MutableMapping
from copy import deepcopy


class Expr:
    """Expression that can be evaluated."""
    
    def __init__(self, str_expr: str):
        self.str_expr = str_expr
    
    def evaluate(self):
        """Evaluate the expression with minimal context"""
        import torch.nn as nn
        import torch
        context = globals().copy()
        context.update(locals())
        context.update({'nn': nn, 'torch': torch, 'th': torch, 'np': __import__('numpy')})
        
        return eval(self.str_expr, context)


class ICfg(MutableMapping):
    """
    Configuration class that can be instantiated.
    
    ICfg represents a configuration that can contain:
    - A class to instantiate (via 'cls' key)
    - Parameters for that class
    - Nested ICfg objects (recursively instantiated)
    - Expr objects (evaluated during instantiation)
    - Regular values (passed through unchanged)
    """

    def __init__(self, **kwargs):
        if 'cls' in kwargs:
            self._class = kwargs.pop('cls')
        else:
            self._class = None
        self._dict = dict(kwargs)

    def instantiate(self):
        """
        Instantiate this configuration.
        
        Returns:
            If _class is set: Instance of the class with processed parameters
            Otherwise: Dictionary of processed parameters
        """
        processed_kwargs = {}
        for key, value in self._dict.items():
            processed_kwargs[key] = self._process_value(value, mode='instantiate')
        
        if self._class:
            resolved_class = self._resolve_class(self._class)
            return resolved_class(**processed_kwargs)
        else:
            return processed_kwargs
    
    def _resolve_class(self, cls_ref):
        """Resolve class reference to actual class object."""
        if isinstance(cls_ref, str):
            module_path, class_name = cls_ref.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        return cls_ref
    
    def to_dict(self):
        """Convert to serializable dictionary."""
        result = dict(self._dict)
        if self._class:
            if isinstance(self._class, type):
                result['cls'] = f"{self._class.__module__}.{self._class.__name__}"
            else:
                result['cls'] = self._class
        return result
    
    @classmethod
    def from_dict(cls, data):
        """Create ICfg from dictionary."""
        return cls(**data)
    
    @staticmethod
    def instantiate_class(cls_ref, *args, **kwargs):
        """
        Instantiate a class with given arguments.
        
        Args:
            cls_ref: Class reference (actual class or string reference)
            *args: Positional arguments for class constructor
            **kwargs: Keyword arguments for class constructor
            
        Returns:
            Instance of the class
        """
        if isinstance(cls_ref, str):
            module_path, class_name = cls_ref.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            resolved_class = getattr(module, class_name)
        else:
            resolved_class = cls_ref
            
        return resolved_class(*args, **kwargs)

    def replace(self, deepcopy_cfg=True, **kwargs):
        """
        Create a new ICfg with updated values.
        
        Args:
            deepcopy_cfg: If True, perform a deep copy of the configuration
            **kwargs: Key-value pairs to update in the copy
            
        Returns:
            A new ICfg object with updated values
        """
        if deepcopy_cfg:
            new_dict = deepcopy(self._dict)
        else:
            new_dict = dict(self._dict)

        new_icfg = ICfg(cls=self._class, **new_dict)
        new_icfg._dict.update(kwargs)
        return new_icfg
    
    def eval(self):
        """
        Recursively evaluate all Expr objects and instantiate all subconfigs with cls.
        
        This method processes the entire configuration tree:
        - Evaluates all Expr objects to their actual values
        - Instantiates all ICfg objects that have a cls attribute
        - Leaves ICfg objects without cls as evaluated ICfg objects
        - Recursively processes nested structures (lists, dicts)
        
        Returns:
            ICfg: A new ICfg with all expressions evaluated and instantiable configs processed
        """
        evaluated_dict = {}
        
        # Process all values in this config
        for key, value in self._dict.items():
            evaluated_dict[key] = self._process_value(value, mode='eval')
        
        # Create new ICfg with evaluated values
        # Keep the class reference but don't instantiate this level yet
        return ICfg(cls=self._class, **evaluated_dict)
    
    def _process_value(self, value, mode='instantiate'):
        """
        Process a value recursively, handling ICfg, Expr, lists, and dicts.
        
        Args:
            value: The value to process
            mode: Either 'instantiate' or 'eval' to control processing behavior
        """
        if isinstance(value, ICfg):
            if mode == 'instantiate':
                return value.instantiate()
            else:  # mode == 'eval'
                if value._class is not None:
                    return value.instantiate()
                else:
                    return value.eval()
        
        elif isinstance(value, Expr):
            return value.evaluate()
        
        elif isinstance(value, list):
            return [self._process_value(item, mode) for item in value]
        
        elif isinstance(value, tuple):
            return tuple(self._process_value(item, mode) for item in value)
        
        elif isinstance(value, dict):
            if 'cls' in value:
                return ICfg(**value).instantiate()
            else:
                return {k: self._process_value(v, mode) for k, v in value.items()}
        
        else:
            return value

    def __setattr__(self, name, value):
        """
        Restrict attribute assignment - only allows updating existing keys.
        Use cfg['key'] = value to add new keys.
        """
        if name.startswith('_'):  # Allow private attributes
            object.__setattr__(self, name, value)
        elif hasattr(self, '_dict') and name in self._dict:
            self._dict[name] = value
        elif not hasattr(self, '_dict'):  # During initialization
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Cannot add new attribute '{name}'. Use cfg['{name}'] = value instead.")
    
    def __getattr__(self, name):
        """Allow accessing dictionary keys as attributes."""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        try:
            return self._dict[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def print_config(self, indent=0, max_depth=None, file=None):
        """Print the configuration in a readable nested format."""
        if max_depth is not None and indent >= max_depth:
            print("  " * indent + "... (max depth reached)", file=file)
            return
        
        if self._class:
            cls_name = (f"{self._class.__module__}.{self._class.__name__}" 
                       if isinstance(self._class, type) else str(self._class))
            print("  " * indent + f"cls: {cls_name}", file=file)
        
        for key, value in self._dict.items():
            self._print_value(key, value, indent, max_depth, file)
    
    def _print_value(self, key, value, indent, max_depth, file=None):
        """Print a single value with appropriate formatting."""
        spaces = "  " * indent
        prefix = f"{spaces}{key}: " if key else spaces
        
        if isinstance(value, ICfg):
            print(f"{prefix}ICfg {{", file=file)
            value.print_config(indent + 1, max_depth, file)
            print(f"{spaces}}}", file=file)
        
        elif isinstance(value, Expr):
            print(f"{prefix}Expr('{value.str_expr}')", file=file)
        
        elif isinstance(value, dict):
            if 'cls' in value:
                print(f"{prefix}dict (with cls) {{", file=file)
                ICfg(**value).print_config(indent + 1, max_depth, file)
                print(f"{spaces}}}", file=file)
            else:
                print(f"{prefix}dict {{", file=file)
                for k, v in value.items():
                    self._print_value(k, v, indent + 1, max_depth, file)
                print(f"{spaces}}}", file=file)
        
        elif isinstance(value, list):
            print(f"{prefix}list [{len(value)} items] [", file=file)
            for i, item in enumerate(value):
                if isinstance(item, (ICfg, Expr, dict)):
                    self._print_value(f"[{i}]", item, indent + 1, max_depth, file)
                else:
                    print(f"  {spaces}[{i}]: {self._format_value(item)}", file=file)
            print(f"{spaces}]", file=file)
        
        elif isinstance(value, tuple):
            print(f"{prefix}tuple ({len(value)} items) (", file=file)
            for i, item in enumerate(value):
                if isinstance(item, (ICfg, Expr, dict)):
                    self._print_value(f"({i})", item, indent + 1, max_depth, file)
                else:
                    print(f"  {spaces}({i}): {self._format_value(item)}", file=file)
            print(f"{spaces})", file=file)
        
        else:
            print(f"{prefix}{self._format_value(value)}", file=file)
    
    def _format_value(self, value):
        """Format a value for display."""
        if isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, type):
            return f"{value.__module__}.{value.__name__}"
        else:
            return str(value)

    # MutableMapping interface
    def __getitem__(self, key): 
        return self._dict[key]
    
    def __setitem__(self, key, value): 
        self._dict[key] = value
    
    def __delitem__(self, key): 
        del self._dict[key]
    
    def __iter__(self): 
        return iter(self._dict)
    
    def __len__(self): 
        return len(self._dict)
