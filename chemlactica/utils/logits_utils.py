import yaml
import os
from typing import List, Any, Dict
from transformers.generation import LogitsProcessor, LogitsProcessorList
from dataclasses import dataclass, field
import importlib
import importlib.util


def import_local_module(module_name, relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(current_dir, relative_path)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not import module {module_name} from {module_path}")

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Execute the module's code and populate the module
    spec.loader.exec_module(module)

    return module


@dataclass
class LogitsProcessorConfig:
    class_name: str
    is_local: str
    module: str
    kwargs: Dict[str, Any]
    path: str = field(default=None)


def instantiate_processors(
    config: List[LogitsProcessorConfig],
) -> List[LogitsProcessor]:
    processors = []
    for processor_config in config:
        if processor_config.is_local:
            module = import_local_module(processor_config.module, processor_config.path)
        else:
            module = importlib.import_module(processor_config.module)
        processor_class = getattr(module, processor_config.class_name)
        processor = processor_class(**processor_config.kwargs)
        processors.append(processor)
    return processors


def load_processor_config(file_path: str) -> List[LogitsProcessorConfig]:
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
        configs = [
            LogitsProcessorConfig(**processor)
            for processor in config_data["logits_processors"]
        ]
        return configs


def get_logits_processors(logits_processors_config_path=None):
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # config_file_path = os.path.join(current_dir, "logit_configs","best_config.yaml")
    if logits_processors_config_path:
        logit_processors_config = load_processor_config(logits_processors_config_path)
        logit_processors = instantiate_processors(logit_processors_config)
        logit_processor_list = LogitsProcessorList(logit_processors)
        return logit_processor_list
    else:
        return None


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, "logit_configs", "best_config.yaml")
    logit_processors_config = load_processor_config(config_file_path)
    logit_processors = instantiate_processors(logit_processors_config)
    logit_processor_list = LogitsProcessorList(logit_processors)
    print(logit_processor_list)
