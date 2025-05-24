from torch.nn import Embedding
from torch.utils.hooks import RemovableHandle


# The following function is adapted from interpret-lm/lm_saliency.py
# https://github.com/kayoyin/interpret-lm/blob/main/lm_saliency.py#L42
def register_embedding_list_hook(
        embedding: Embedding,
        embeddings_list: list
    ) -> RemovableHandle:
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach())
    embedding_layer = embedding
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

# The following function is adapted from interpret-lm/lm_saliency.py
# https://github.com/kayoyin/interpret-lm/blob/main/lm_saliency.py#L49
def register_embedding_gradient_hooks(
        embedding: Embedding,
        embeddings_gradients: list
    ) -> RemovableHandle:
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu())
    embedding_layer = embedding
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook