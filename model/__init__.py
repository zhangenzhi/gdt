from model.hvit import hvit_b, hvit_l, hvit_xl

# 模型工厂映射字典
_model_factory = {
    "hvit_b": hvit_b,
    "hvit_l": hvit_l,
    "hvit_xl": hvit_xl,
}

def create_model(model_type, **kwargs):
    """
    模型创建工厂函数。
    参数:
        model_type: 字符串，例如 'hvit_b', 'hvit_l', 'hvit_xl'
        **kwargs: 传递给模型构造函数的参数 (img_size, patch_size 等)
    """
    if model_type not in _model_factory:
        available = ", ".join(_model_factory.keys())
        raise ValueError(f"不支持的模型类型: {model_type}。可选类型包括: {available}")
    
    # 获取对应的工厂函数并实例化
    model_func = _model_factory[model_type]
    return model_func(**kwargs)