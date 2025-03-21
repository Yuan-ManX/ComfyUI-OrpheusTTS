from .OrpheusTTS_nodes import SingleTextGeneration, LongTextGeneration

NODE_CLASS_MAPPINGS = {
    "Single Text Generation": SingleTextGeneration,
    "Long Text Generation": LongTextGeneration
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SingleTextGeneration": "Single Text Generation",
    "LongTextGeneration": "Long Text Generation"
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
