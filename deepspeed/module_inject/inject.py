import copy
import torch
from deepspeed.ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig


def module_inject(layer_obj,
                  model,
                  config,
                  micro_batch_size,
                  max_seq_length,
                  seed,
                  preln,
                  fp16=True):
    for name, child in model.named_children():
        if isinstance(child, layer_obj):
            print('REPLACING BertLayer')

            cuda_config = DeepSpeedTransformerConfig(
                batch_size=micro_batch_size,
                max_seq_length=max_seq_length,
                hidden_size=config.hidden_size,
                heads=config.num_attention_heads,
                attn_dropout_ratio=config.attention_probs_dropout_prob,
                hidden_dropout_ratio=config.hidden_dropout_prob,
                num_hidden_layers=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                seed=seed,
                fp16=fp16,
                pre_layer_norm=preln)

            new_module = DeepSpeedTransformerLayer(cuda_config)

            # copy relevant state from child -> new module
            qw = child.attention.self.query.weight
            qb = child.attention.self.query.bias
            kw = child.attention.self.key.weight
            kb = child.attention.self.key.bias
            vw = child.attention.self.value.weight
            vb = child.attention.self.value.bias

            qkvw = torch.cat((qw, kw, vw), 0)
            qkvb = torch.cat((qb, kb, vb), 0)

            new_module.attn_qkvw.data = qkvw
            new_module.attn_qkvb.data = qkvb
            new_module.attn_ow.data = child.attention.output.dense.weight
            new_module.attn_ob.data = child.attention.output.dense.bias
            if preln:
                attention_layerNorm = child.PostAttentionLayerNorm
            else:
                attention_layerNorm = child.attention.output.LayerNorm
            new_module.attn_nw.data = attention_layerNorm.weight
            new_module.attn_nb.data = attention_layerNorm.bias
            if preln:
                intermediate_FF = child.intermediate.dense_act
            else:
                intermediate_FF = child.intermediate.dense
            new_module.inter_w.data = intermediate_FF.weight
            new_module.inter_b.data = intermediate_FF.bias
            new_module.output_w.data = child.output.dense.weight
            new_module.output_b.data = child.output.dense.bias
            if preln:
                transformer_LayerNorm = child.PreAttentionLayerNorm
            else:
                transformer_LayerNorm = child.output.LayerNorm
            new_module.norm_w.data = transformer_LayerNorm.weight
            new_module.norm_b.data = transformer_LayerNorm.bias

            setattr(model, name, copy.deepcopy(new_module))

        else:
            module_inject(layer_obj,
                          child,
                          config,
                          micro_batch_size,
                          max_seq_length,
                          seed,
                          preln,
                          fp16)

    return model
