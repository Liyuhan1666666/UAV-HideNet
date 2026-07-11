_base_ = [
    "../_base_/common.py",
    "../_base_/train.py",
    "../_base_/test.py",
]

has_test = True
deterministic = True
use_custom_worker_init = False
model_name = "UAVHideNet"


train = dict(
    batch_size=6,
    num_workers=4,
    use_amp=True,
    num_epochs=100,
    epoch_based=True,
    lr=0.001,
    ema=dict(
            enable=True,        # 开启EMA
            decay=0.999,        # 衰减系数（推荐0.999/0.9999，越大越稳定）
            force_cpu=False,    # 不强制放CPU（GPU足够时）
            keep_original_test=True,  # 测试时同时验证原始模型和EMA模型
    ),
    optimizer=dict(
        mode="adamw",           # 使用 AdamW
        set_to_none=True,
        group_mode="finetune",
        cfg=dict(
            weight_decay=1e-4,
            betas=(0.9, 0.999), # Adam 的动量参数
            eps=1e-8,            # 数值稳定性
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=500,
            initial_coef=0.01,
            mode="linear",
        ),
        mode="cos",
        cfg=dict(
            min_coef=0.001,
        ),
    ),
)

test = dict(
    batch_size=8,
    num_workers=4,
    show_bar=False,
)




datasets = dict(
    train=dict(
        dataset_type="msi_cod_tr",
        shape=dict(h=384, w=384),
        path=["msi_cod_tr"],
        interp_cfg=dict(),
    ),
    test=dict(
        dataset_type="msi_cod_te",
        shape=dict(h=384, w=384),
        path=["msi_cod_te"],
        interp_cfg=dict(),
    ),
)
