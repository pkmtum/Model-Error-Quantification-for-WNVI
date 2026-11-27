def fields_plot_prep(fields):
    if (fields.dim() == 0) or (fields.dim() == 1):
        return fields
    # flatten if still not flattened
    if fields.size(-1) == fields.size(-2):
        fields = fields.flatten(start_dim=-2)
    if fields.size(0) < fields.size(-1):
        fields = fields.transpose(0, -1)
    return fields