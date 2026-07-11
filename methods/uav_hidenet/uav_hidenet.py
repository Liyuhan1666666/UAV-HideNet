import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import timm

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops import cus_sample



# 工具函数
def get_coef(iter_percentage, method):
    """
    根据训练进度计算不确定性感知损失（UAL）的系数。
    参数：
        iter_percentage: float, 当前迭代占总迭代的比例
        method: str, 调度方法，可选 'linear' 或 'cos'
    返回：
        float: 系数
    """
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits, seg_gts):
    """
    计算不确定性感知损失（Uncertainty-Aware Loss）。
    参数：
        seg_logits: 模型预测的 logits
        seg_gts: 真实标签
    返回：
        float: 损失值
    """
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


# 基础模块
class ChannelEmphasisConv(nn.Module):
    """通道强调卷积：通过可学习的门控抑制冗余通道"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        gate_weight = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        return x * gate_weight


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch_att = self.channel_att(x)
        x_ch = x * ch_att
        max_pool = torch.max(x_ch, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_ch, dim=1, keepdim=True)
        spatial_feat = torch.cat([max_pool, avg_pool], dim=1)
        sp_att = self.spatial_att(spatial_feat)
        return x_ch * sp_att


class LearnableChannelScaling(nn.Module):
    """可学习通道缩放层，用于稳定训练"""
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return x * self.scale


class DReAM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.emphasis_l = ChannelEmphasisConv(in_dim, in_dim)
        self.emphasis_m = ChannelEmphasisConv(in_dim, in_dim)
        self.emphasis_s = ChannelEmphasisConv(in_dim, in_dim)
        self.collaborative_attention = ChannelSpatialAttention(in_dim * 3)
        self.channel_scaling = LearnableChannelScaling(in_dim * 3)
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_dim * 3, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Identity()  # 以中间尺度 m 作为残差

    def forward(self, l, m, s, return_feats=False):
        """
        输入三个尺度的特征图（大、中、小），输出融合后的特征。
        参数：
            l: 大尺度特征 (B, C, H_l, W_l)
            m: 中尺度特征 (B, C, H_m, W_m)
            s: 小尺度特征 (B, C, H_s, W_s)
            return_feats: 是否返回中间特征用于分析
        返回：
            若 return_feats=False: 融合后的特征 (B, C, H_m, W_m)
            若 return_feats=True: (out, dict) 包含中间增强特征
        """
        tgt_size = m.shape[2:]
        l_aligned = F.interpolate(l, size=tgt_size, mode='bilinear', align_corners=False)
        s_aligned = F.interpolate(s, size=tgt_size, mode='bilinear', align_corners=False)

        l_enhanced = self.emphasis_l(l_aligned)
        m_enhanced = self.emphasis_m(m)
        s_enhanced = self.emphasis_s(s_aligned)

        cat_feat = torch.cat([l_enhanced, m_enhanced, s_enhanced], dim=1)
        cat_feat = self.channel_scaling(cat_feat)
        cat_feat = self.collaborative_attention(cat_feat)
        out = self.out_proj(cat_feat)
        out = out + self.residual(m)  # 残差连接

        if return_feats:
            return out, {
                'l_enhanced': l_enhanced,
                'm_enhanced': m_enhanced,
                's_enhanced': s_enhanced,
                'attention_out': cat_feat
            }
        return out


class TransLayer(nn.Module):
    """将多级编码器特征统一到相同通道数"""
    def __init__(self, out_c):
        super().__init__()
        self.c5_down = nn.Sequential(ConvBNReLU(2048, out_c, 3, 1, 1))
        self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

    def forward(self, xs):
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return c5, c4, c3, c2, c1


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class HMU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []

        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(3, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(2, dim=1))

        out = torch.cat([o[0] for o in outs], dim=1)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)


# ==================== 主网络 ====================
@MODELS.register()
class UAVHideNet(BasicModelClass):
    """UAV-HIDENET 基础版本，不使用梯度检查点，适合显存充足时训练"""
    def __init__(self):
        super().__init__()
        # 共享编码器
        self.shared_encoder = timm.create_model("resnet50", pretrained=True, features_only=True)
        # 跨尺度特征转换层
        self.translayer = TransLayer(out_c=64)
        # RFB 模块增强最高层特征
        self.rfb_c5 = RFB_modified(in_channel=64, out_channel=64)

        # 多尺度融合模块（使用 DReAM）
        self.merge_layers = nn.ModuleList([DReAM(in_dim=in_c) for in_c in (64, 64, 64, 64, 64)])

        # 解码器中的 HMU 模块
        self.d5 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d4 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d3 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d2 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d1 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        # 最终输出层
        self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
        self.out_layer_01 = nn.Conv2d(32, 1, 1)

    def encoder_translayer(self, x):
        """编码器 + 转换层，输出统一通道数的多级特征"""
        en_feats = self.shared_encoder(x)
        trans_feats = self.translayer(en_feats)
        return trans_feats

    def body(self, l_scale, m_scale, s_scale):
        """主体前向传播：处理三个尺度的输入，返回分割预测"""
        # 提取三个尺度的特征
        l_trans_feats = self.encoder_translayer(l_scale)
        m_trans_feats = self.encoder_translayer(m_scale)
        s_trans_feats = self.encoder_translayer(s_scale)

        l_trans_feats = list(l_trans_feats)
        m_trans_feats = list(m_trans_feats)
        s_trans_feats = list(s_trans_feats)

        # 对最高层特征进行 RFB 增强
        l_trans_feats[0] = self.rfb_c5(l_trans_feats[0])
        m_trans_feats[0] = self.rfb_c5(m_trans_feats[0])
        s_trans_feats[0] = self.rfb_c5(s_trans_feats[0])

        # 多尺度融合
        feats = []
        for l, m, s, layer in zip(l_trans_feats, m_trans_feats, s_trans_feats, self.merge_layers):
            siu_outs = layer(l=l, m=m, s=s)
            feats.append(siu_outs)

        # 解码器逐步上采样融合
        x = self.d5(feats[0])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + feats[1])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + feats[2])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + feats[3])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + feats[4])
        x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        return dict(seg=logits)

    def train_forward(self, data, **kwargs):
        """训练时前向传播，返回预测、损失和损失字符串"""
        assert not {"image1.5", "image1.0", "image0.5", "mask"}.difference(set(data)), set(data)

        output = self.body(
            l_scale=data["image1.5"],
            m_scale=data["image1.0"],
            s_scale=data["image0.5"],
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        return dict(sal=output["seg"].sigmoid()), loss, loss_str

    def test_forward(self, data, **kwargs):
        """测试时前向传播，返回分割 logits"""
        output = self.body(
            l_scale=data["image1.5"],
            m_scale=data["image1.0"],
            s_scale=data["image0.5"],
        )
        return output["seg"]

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
        """计算总损失（BCE + 不确定性感知损失）"""
        ual_coef = get_coef(iter_percentage, method)

        losses = []
        loss_str = []
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])

            sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")

            ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
            ual_loss *= ual_coef
            losses.append(ual_loss)
            loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
        return sum(losses), " ".join(loss_str)

    def get_grouped_params(self):
        """分组参数，用于不同学习率策略"""
        param_groups = {}
        for name, param in self.named_parameters():
            if name.startswith("shared_encoder.layer"):
                param_groups.setdefault("pretrained", []).append(param)
            elif name.startswith("shared_encoder."):
                param_groups.setdefault("fixed", []).append(param)
            elif "rfb_c5" in name:
                param_groups.setdefault("retrained", []).append(param)
            else:
                param_groups.setdefault("retrained", []).append(param)
        return param_groups


@MODELS.register()
class UAVHideNet_CK(UAVHideNet):
    """
    UAV-HIDENET 的内存节省版本，使用梯度检查点（checkpoint）技术。
    继承自 UAV-HIDENET，重写 body 方法，将编码器、转换层、融合层、解码器包裹在 checkpoint 中。
    适用于显存受限的场景，但会额外增加计算时间。
    """
    def __init__(self):
        super().__init__()
        # 引入一个虚拟参数，用于满足 checkpoint 对输入梯度依赖的要求
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def encoder(self, x, dummy_arg=None):
        """编码器，使用 checkpoint 时要求输入必须包含所有依赖的 tensor"""
        assert dummy_arg is not None
        x0, x1, x2, x3, x4 = self.shared_encoder(x)
        return x0, x1, x2, x3, x4

    def trans(self, x0, x1, x2, x3, x4):
        """特征转换层，将多级特征统一通道数"""
        x5, x4, x3, x2, x1 = self.translayer([x0, x1, x2, x3, x4])
        return x5, x4, x3, x2, x1

    def decoder(self, x5, x4, x3, x2, x1):
        """解码器，逐步上采样融合"""
        x = self.d5(x5)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + x4)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + x3)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + x2)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + x1)
        x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        return logits

    def body(self, l_scale, m_scale, s_scale):
        """主体前向传播，所有可能耗显存的部分均使用 checkpoint"""
        # 编码器部分使用 checkpoint
        l_trans_feats = checkpoint(self.encoder, l_scale, self.dummy)
        m_trans_feats = checkpoint(self.encoder, m_scale, self.dummy)
        s_trans_feats = checkpoint(self.encoder, s_scale, self.dummy)

        # 特征转换层使用 checkpoint
        l_trans_feats = checkpoint(self.trans, *l_trans_feats)
        m_trans_feats = checkpoint(self.trans, *m_trans_feats)
        s_trans_feats = checkpoint(self.trans, *s_trans_feats)

        # 多尺度融合模块使用 checkpoint
        feats = []
        for layer_idx, (l, m, s) in enumerate(zip(l_trans_feats, m_trans_feats, s_trans_feats)):
            siu_outs = checkpoint(self.merge_layers[layer_idx], l, m, s)
            feats.append(siu_outs)

        # 解码器使用 checkpoint
        logits = checkpoint(self.decoder, *feats)
        return dict(seg=logits)


