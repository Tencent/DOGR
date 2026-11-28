"""Small offline regression tests; no pretrained checkpoint is downloaded."""
import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


class OptionalAscendTests(unittest.TestCase):
    def load_backend(self, error=None, module=None):
        path = Path(__file__).resolve().parents[1] / "llava" / "accelerators.py"
        spec = importlib.util.spec_from_file_location("optional_ascend_test", path)
        loaded = importlib.util.module_from_spec(spec)
        original = builtins.__import__

        def controlled_import(name, *args, **kwargs):
            if name == "torch_npu":
                if error:
                    raise error
                return module
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=controlled_import):
            spec.loader.exec_module(loaded)
        return loaded

    def test_missing_ascend_is_optional(self):
        loaded = self.load_backend(ModuleNotFoundError("missing", name="torch_npu"))
        self.assertIsNone(loaded.torch_npu)

    def test_available_ascend_is_preserved(self):
        backend = object()
        self.assertIs(self.load_backend(module=backend).torch_npu, backend)

    def test_broken_installed_backend_is_not_hidden(self):
        with self.assertRaises(ModuleNotFoundError):
            self.load_backend(ModuleNotFoundError("missing vendor dependency", name="vendor_runtime"))

    def test_cpu_qwen_forward_without_ascend(self):
        import torch
        from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

        config = LlavaQwenConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=1, num_attention_heads=4,
                                num_key_value_heads=2, max_position_embeddings=64)
        model = LlavaQwenForCausalLM(config).eval()
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            result = model(input_ids=ids)
        self.assertEqual(tuple(result.logits.shape), (1, 3, 64))
        self.assertTrue(torch.isfinite(result.logits).all().item())

    def test_vision_tower_follows_model_placement(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from llava.model.builder import load_pretrained_model

        tokenizer = MagicMock()
        tokenizer.__len__.return_value = 64
        vision = MagicMock(is_loaded=True)
        model = MagicMock()
        model.device = torch.device("cpu")
        model.dtype = torch.float32
        model.config = SimpleNamespace(mm_use_im_patch_token=False, mm_use_im_start_end=False,
                                       max_position_embeddings=64)
        model.get_vision_tower.return_value = vision
        with patch("llava.model.builder.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("llava.model.builder.LlavaQwenForCausalLM.from_pretrained", return_value=model):
            load_pretrained_model("unused-local-path", None, "llava_qwen", device_map="cpu", attn_implementation="sdpa")
        vision.to.assert_called_once_with(device=torch.device("cpu"), dtype=torch.float32)
        self.assertIs(model.tokenizer, tokenizer)

    def test_multimodal_forward_keeps_device(self):
        import torch
        from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
        from llava.constants import IMAGE_TOKEN_INDEX

        class TinyVision(torch.nn.Module):
            def forward(self, images):
                return torch.nn.functional.adaptive_avg_pool2d(images, (2, 2)).flatten(2).transpose(1, 2)

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                config = LlavaQwenConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=1, num_attention_heads=4,
                                        num_key_value_heads=2, max_position_embeddings=64)
                config.mm_patch_merge_type = "flat"
                model = LlavaQwenForCausalLM(config)
                model.model.vision_tower = TinyVision()
                model.model.mm_projector = torch.nn.Linear(3, 32)
                model = model.to(device).eval()
                ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 3]], device=device)
                images = torch.randn(1, 3, 4, 4, device=device)
                with torch.no_grad():
                    result = model(input_ids=ids, images=images)
                self.assertEqual(tuple(result.logits.shape), (1, 6, 64))
                self.assertEqual(result.logits.device.type, device)
                self.assertTrue(torch.isfinite(result.logits).all().item())


if __name__ == "__main__":
    unittest.main()
