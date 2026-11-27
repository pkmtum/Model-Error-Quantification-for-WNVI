def input_dict_for_ParameterToField(
    transformerInputKeys: str | list[str] | tuple[str, ...], **kwargs
) -> dict:
    input_keys = list(transformerInputKeys)
    parameters = {}
    if input_keys and input_keys[0] in kwargs:
        parameters["parameter"] = kwargs[input_keys[0]]
    for k in input_keys[1:]:
        if k in kwargs:
            parameters[k] = kwargs[k]

    return parameters
