import pytest
import torch.nn as nn
from masterthesis.icfg import ICfg, Expr


class MockClass:
    """Mock class for testing"""
    def __init__(self, **kwargs):
        # Accept all kwargs for flexible testing
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestExpr:
    """Test cases for the Expr class"""
    
    def test_simple_expression(self):
        expr = Expr("1 + 2")
        assert expr.evaluate() == 3
    
    def test_torch_nn_expression(self):
        expr = Expr("nn.ReLU()")
        result = expr.evaluate()
        assert isinstance(result, nn.ReLU)
    
    def test_torch_nn_with_params(self):
        expr = Expr("nn.LeakyReLU(0.1)")
        result = expr.evaluate()
        assert isinstance(result, nn.LeakyReLU)
        assert result.negative_slope == 0.1
    
    def test_complex_expression(self):
        expr = Expr("torch.tensor([1, 2, 3]).sum().item()")
        assert expr.evaluate() == 6


class TestICfg:
    """Test cases for the ICfg class"""
    
    def test_init_without_cls(self):
        cfg = ICfg(param1="value1", param2=42)
        assert cfg._class is None
        assert cfg['param1'] == "value1"
        assert cfg['param2'] == 42
    
    def test_init_with_cls(self):
        cfg = ICfg(cls=MockClass, param1="value1")
        assert cfg._class == MockClass
        assert cfg['param1'] == "value1"
    
    def test_mutable_mapping_interface(self):
        cfg = ICfg(param1="value1")
        
        # Test __getitem__ and __setitem__
        assert cfg['param1'] == "value1"
        cfg['param2'] = "value2"
        assert cfg['param2'] == "value2"
        
        # Test __delitem__
        del cfg['param1']
        assert 'param1' not in cfg
        
        # Test __iter__ and __len__
        assert len(cfg) == 1
        assert list(cfg) == ['param2']
    
    def test_resolve_class_direct_reference(self):
        cfg = ICfg(cls=MockClass)
        resolved = cfg._resolve_class(MockClass)
        assert resolved == MockClass
    
    def test_resolve_class_string_reference(self):
        cfg = ICfg()
        resolved = cfg._resolve_class("torch.nn.ReLU")
        assert resolved == nn.ReLU
    
    def test_instantiate_without_class(self):
        cfg = ICfg(param1="value1", param2=42)
        result = cfg.instantiate()
        assert result == {"param1": "value1", "param2": 42}
    
    def test_instantiate_with_class(self):
        cfg = ICfg(cls=MockClass, param1="value1", param2=42)
        result = cfg.instantiate()
        assert isinstance(result, MockClass)
        assert result.param1 == "value1"
        assert result.param2 == 42
    
    def test_instantiate_with_string_class(self):
        cfg = ICfg(cls="torch.nn.ReLU")
        result = cfg.instantiate()
        assert isinstance(result, nn.ReLU)
    
    def test_process_value_expr(self):
        cfg = ICfg()
        expr = Expr("nn.ReLU()")
        result = cfg._process_value(expr)
        assert isinstance(result, nn.ReLU)
    
    def test_process_value_nested_icfg(self):
        nested_cfg = ICfg(cls=MockClass, param1="nested_value")
        cfg = ICfg(cls=MockClass, nested=nested_cfg)
        result = cfg.instantiate()
        
        assert isinstance(result, MockClass)
        assert isinstance(result.nested, MockClass)
        assert result.nested.param1 == "nested_value"
    
    def test_process_value_list(self):
        cfg = ICfg()
        test_list = [
            Expr("nn.ReLU()"),
            ICfg(cls=MockClass, param1="test"),
            "plain_string"
        ]
        result = cfg._process_value(test_list)
        
        assert isinstance(result[0], nn.ReLU)
        assert isinstance(result[1], MockClass)
        assert result[1].param1 == "test"
        assert result[2] == "plain_string"
    
    def test_process_value_dict_with_cls(self):
        cfg = ICfg()
        test_dict = {"cls": MockClass, "param1": "test_value"}
        result = cfg._process_value(test_dict)
        
        assert isinstance(result, MockClass)
        assert result.param1 == "test_value"
    
    def test_process_value_dict_without_cls(self):
        cfg = ICfg()
        test_dict = {"key1": "value1", "key2": Expr("1 + 1")}
        result = cfg._process_value(test_dict)
        
        assert result["key1"] == "value1"
        assert result["key2"] == 2
    
    def test_complex_nested_configuration(self):
        """Test a complex nested configuration similar to the experiment_cfg"""
        config_dict = {
            'cls': MockClass,
            'param1': 'top_level',
            'trainer': {
                'cls': MockClass,
                'param1': 'trainer_param',
                'optimizer': {
                    'cls': 'torch.nn.ReLU',  # Using string reference
                }
            },
            'activations': [
                Expr('nn.ReLU()'),
                Expr('nn.LeakyReLU(0.1)'),
            ]
        }
        
        cfg = ICfg(**config_dict)
        result = cfg.instantiate()
        
        assert isinstance(result, MockClass)
        assert result.param1 == 'top_level'
        assert isinstance(result.trainer, MockClass)
        assert result.trainer.param1 == 'trainer_param'
        assert isinstance(result.trainer.optimizer, nn.ReLU)
        assert isinstance(result.activations[0], nn.ReLU)
        assert isinstance(result.activations[1], nn.LeakyReLU)
    
    def test_to_dict_with_class_reference(self):
        cfg = ICfg(cls=MockClass, param1="value1")
        result = cfg.to_dict()
        
        expected_cls = f"{MockClass.__module__}.{MockClass.__name__}"
        assert result['cls'] == expected_cls
        assert result['param1'] == "value1"
    
    def test_to_dict_with_string_reference(self):
        cfg = ICfg(cls="torch.nn.ReLU", param1="value1")
        result = cfg.to_dict()
        
        assert result['cls'] == "torch.nn.ReLU"
        assert result['param1'] == "value1"
    
    def test_from_dict(self):
        data = {'cls': MockClass, 'param1': 'test_value'}
        cfg = ICfg.from_dict(data)
        
        assert cfg._class == MockClass
        assert cfg['param1'] == 'test_value'
    
    def test_roundtrip_serialization(self):
        """Test that we can serialize and deserialize configurations"""
        original_cfg = ICfg(cls=MockClass, param1="test", param2=42)
        
        # Serialize to dict
        serialized = original_cfg.to_dict()
        
        # Deserialize back
        restored_cfg = ICfg.from_dict(serialized)
        
        # Should be functionally equivalent (both instantiate to same result)
        original_result = original_cfg.instantiate()
        restored_result = restored_cfg.instantiate()
        
        assert type(original_result) == type(restored_result)
        assert original_result.param1 == restored_result.param1
        assert original_result.param2 == restored_result.param2
    
    def test_error_on_invalid_string_class(self):
        cfg = ICfg(cls="nonexistent.module.Class")
        with pytest.raises((ImportError, AttributeError)):
            cfg.instantiate()
    
    def test_error_on_invalid_expr(self):
        expr = Expr("undefined_function()")
        with pytest.raises(NameError):
            expr.evaluate()
    
    def test_instantiate_class_static_method(self):
        """Test the static instantiate_class method"""
        # Test with direct class reference
        result = ICfg.instantiate_class(MockClass, param1="test", param2=42)
        assert isinstance(result, MockClass)
        assert result.param1 == "test"
        assert result.param2 == 42
        
        # Test with string class reference
        result = ICfg.instantiate_class("torch.nn.ReLU")
        assert isinstance(result, nn.ReLU)
        
        # Test with string class reference and parameters
        result = ICfg.instantiate_class("torch.nn.LeakyReLU", negative_slope=0.1)
        assert isinstance(result, nn.LeakyReLU)
        assert result.negative_slope == 0.1
    
    def test_replace_functionality(self):
        """Test the replace method creates a new ICfg with updated values"""
        original_cfg = ICfg(cls=MockClass, param1="original", param2=42)
        
        # Create a new config with replaced values
        new_cfg = original_cfg.replace(param1="updated", param3="new")
        
        # Original should be unchanged
        assert original_cfg['param1'] == "original"
        assert original_cfg['param2'] == 42
        assert 'param3' not in original_cfg
        
        # New config should have updated values
        assert new_cfg['param1'] == "updated"
        assert new_cfg['param2'] == 42  # unchanged
        assert new_cfg['param3'] == "new"
        assert new_cfg._class == MockClass
    
    def test_attribute_access(self):
        """Test that attributes can be accessed with dot notation"""
        cfg = ICfg(cls=MockClass, param1="value1", param2=42)
        
        # Should be able to read attributes
        assert cfg.param1 == "value1"
        assert cfg.param2 == 42
    
    def test_restricted_attribute_assignment(self):
        """Test that attribute assignment only works for existing keys"""
        cfg = ICfg(cls=MockClass, param1="value1", param2=42)
        
        # Should be able to update existing attributes
        cfg.param1 = "updated"
        assert cfg.param1 == "updated"
        assert cfg['param1'] == "updated"
        
        # Should NOT be able to add new attributes with dot notation
        with pytest.raises(AttributeError, match="Cannot add new attribute 'new_param'"):
            cfg.new_param = "should_fail"
        
        # But should be able to add with bracket notation
        cfg['new_param'] = "works"
        assert cfg['new_param'] == "works"
        
        # And then should be able to update with dot notation
        cfg.new_param = "updated_via_dot"
        assert cfg.new_param == "updated_via_dot"
    
    def test_attribute_access_nonexistent(self):
        """Test that accessing nonexistent attributes raises AttributeError"""
        cfg = ICfg(cls=MockClass, param1="value1")
        
        with pytest.raises(AttributeError, match="object has no attribute 'nonexistent'"):
            _ = cfg.nonexistent
    
    def test_print_config(self, capsys):
        """Test the print_config method"""
        config_dict = {
            'cls': MockClass,
            'param1': 'top_level',
            'trainer': {
                'cls': 'torch.nn.ReLU',
                'param1': 'trainer_param',
            },
            'activations': [
                Expr('nn.ReLU()'),
                'plain_string',
            ],
            'nested_icfg': ICfg(cls=MockClass, nested_param='test')
        }
        
        cfg = ICfg(**config_dict)
        cfg.print_config(max_depth=3)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that key elements are printed
        assert 'cls: test.test_icfg.MockClass' in output
        assert 'param1: \'top_level\'' in output
        assert 'trainer: dict (with cls)' in output
        assert 'activations: list [2 items]' in output
        assert 'Expr(\'nn.ReLU()\')' in output
        assert 'nested_icfg: ICfg' in output


if __name__ == "__main__":
    pytest.main(["-v", __file__])
