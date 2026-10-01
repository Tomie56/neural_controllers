from predictive_memory_localization.common import resolve_model_layers


def generate_on_text(model, tokenizer, input_text, **kwargs):
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    outputs = model.generate(
        **inputs,
        **kwargs,
    )
    return tokenizer.decode(outputs[0])


def hook_model(model, directions, layers_to_control, control_coef, component_idx=0):
    hooks = {}
    model_layers, _ = resolve_model_layers(model)
    for layer_idx in layers_to_control:
        control_vec = directions[layer_idx][component_idx]
        if len(control_vec.shape) == 1:
            control_vec = control_vec.reshape(1, 1, -1)

        block = model_layers[layer_idx]

        def block_hook(module, input, output, control_vec=control_vec, control_coef=control_coef):
            is_tuple = isinstance(output, tuple)
            new_output = output[0] if is_tuple else output
            new_output = new_output + control_coef * control_vec.to(
                dtype=new_output.dtype,
                device=new_output.device,
            )
            if is_tuple:
                new_output = (new_output,) + output[1:]
            return new_output

        hooks[layer_idx] = block.register_forward_hook(block_hook)

    return hooks


def clear_hooks(hooks) -> None:
    for hook_handle in hooks.values():
        hook_handle.remove()
