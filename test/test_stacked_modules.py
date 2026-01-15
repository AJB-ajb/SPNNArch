import pytest
import torch
from masterthesis.stacked_torch_modules import StackedLinear


class TestStackedLinear:
    
    @pytest.fixture
    def layer_params(self):
        return {
            "n_instances": 3,
            "n_in": 4,
            "n_out": 5,
            "batch_size": 2
        }
    
    @pytest.fixture
    def stacked_linear(self, layer_params):
        return StackedLinear(
            n_instances=layer_params["n_instances"],
            n_in=layer_params["n_in"],
            n_out=layer_params["n_out"]
        )
    
    def test_layer_initialization(self, stacked_linear, layer_params):
        assert stacked_linear.weight.shape == (
            layer_params["n_instances"],
            layer_params["n_in"],
            layer_params["n_out"]
        )
        assert stacked_linear.bias.shape == (
            layer_params["n_instances"],
            layer_params["n_out"]
        )
    
    def test_forward_3d_input(self, stacked_linear, layer_params):
        input_tensor = torch.randn(
            layer_params["batch_size"],
            layer_params["n_instances"],
            layer_params["n_in"]
        )
        
        output = stacked_linear(input_tensor)
        
        assert output.shape == (
            layer_params["batch_size"],
            layer_params["n_instances"],
            layer_params["n_out"]
        )
        
        instance_idx, batch_idx = 0, 0
        
        manual_output = torch.matmul(
            input_tensor[batch_idx, instance_idx], 
            stacked_linear.weight[instance_idx]
        ) + stacked_linear.bias[instance_idx]
        
        assert torch.allclose(
            output[batch_idx, instance_idx], 
            manual_output,
            rtol=1e-4
        )
    
    def test_forward_2d_input(self, stacked_linear, layer_params):
        input_tensor = torch.randn(
            layer_params["batch_size"],
            layer_params["n_in"]
        )
        
        output = stacked_linear(input_tensor)
        
        assert output.shape == (
            layer_params["batch_size"],
            layer_params["n_instances"],
            layer_params["n_out"]
        )
        
        instance_idx, batch_idx = 0, 0
        
        manual_output = torch.matmul(
            input_tensor[batch_idx], 
            stacked_linear.weight[instance_idx]
        ) + stacked_linear.bias[instance_idx]
        
        assert torch.allclose(
            output[batch_idx, instance_idx], 
            manual_output,
            rtol=1e-4
        )
    
    def test_no_bias(self, layer_params):
        layer = StackedLinear(
            n_instances=layer_params["n_instances"],
            n_in=layer_params["n_in"],
            n_out=layer_params["n_out"],
            bias=False
        )
        
        assert layer.bias is None
        
        input_tensor = torch.randn(
            layer_params["batch_size"],
            layer_params["n_instances"],
            layer_params["n_in"]
        )
        
        output = layer(input_tensor)
        
        assert output.shape == (
            layer_params["batch_size"],
            layer_params["n_instances"],
            layer_params["n_out"]
        )
        
        input_tensor = torch.randn(
            layer_params["batch_size"],
            layer_params["n_in"]
        )
        
        output = layer(input_tensor)
        
        assert output.shape == (
            layer_params["batch_size"],
            layer_params["n_instances"],
            layer_params["n_out"]
        )
    
    def test_gpu_compatibility(self, stacked_linear, layer_params):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU test")
        
        device = torch.device("cuda")
        stacked_linear = stacked_linear.to(device)
        
        input_tensor = torch.randn(
            layer_params["batch_size"],
            layer_params["n_instances"],
            layer_params["n_in"],
            device=device
        )
        
        output = stacked_linear(input_tensor)
        
        assert output.shape == (
            layer_params["batch_size"],
            layer_params["n_instances"],
            layer_params["n_out"]
        )
        assert output.device == device
        
        input_tensor = torch.randn(
            layer_params["batch_size"],
            layer_params["n_in"],
            device=device
        )
        
        output = stacked_linear(input_tensor)
        
        assert output.shape == (
            layer_params["batch_size"],
            layer_params["n_instances"],
            layer_params["n_out"]
        )
        assert output.device == device
    
    def test_gradients(self, stacked_linear, layer_params):
        input_tensor = torch.randn(
            layer_params["batch_size"],
            layer_params["n_instances"],
            layer_params["n_in"],
            requires_grad=True
        )
        
        output = stacked_linear(input_tensor)
        loss = output.mean()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert stacked_linear.weight.grad is not None
        assert stacked_linear.bias.grad is not None
        
        stacked_linear.zero_grad()
        input_tensor.grad = None
        
        input_tensor = torch.randn(
            layer_params["batch_size"],
            layer_params["n_in"],
            requires_grad=True
        )
        
        output = stacked_linear(input_tensor)
        loss = output.mean()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert stacked_linear.weight.grad is not None
        assert stacked_linear.bias.grad is not None

if __name__ == "__main__":
    pytest.main(["-v", __file__])