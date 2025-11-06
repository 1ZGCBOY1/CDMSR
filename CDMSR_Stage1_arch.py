import CDMSR.archs.common as common
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn import functional as F
from einops import rearrange
import math  # 添加math导入
from torch.nn import init  # 添加init导入
from CDMSR.archs.func import window_partitionx, window_reversex
# 从mamba_arch.py中导入必要的类和函数
# 使用绝对导入
from CDMSR.archs.mamba_arch import ESSM, PatchEmbed, PatchUnEmbed


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# todo 添加FCM和MPU类
class frequency_selection(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, window_size=None, bias=False):
        super(frequency_selection, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        # dim = out_channel
        hid_dim = dim * dw
        # print(dim, hid_dim)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.freq_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm

    def forward(self, x): # (4,96,64,64)
        _, _, H, W = x.shape
        freq_weight = self.freq_attention(x)
        x_weighted = x * freq_weight

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x_weighted, batch_list = window_partitionx(x_weighted, self.window_size)

        y = torch.fft.rfft2(x_weighted, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)

        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)

        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class MPU(nn.Module):
    """Multi-scale Processing Unit - 多尺度处理单元"""

    def __init__(self, dim, bias=False):
        super(MPU, self).__init__()

        # 深度可分离卷积分支 (3x3)
        self.dw_conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.pw_conv3x3 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 点卷积分支 (5x5通过1x1实现)
        self.pw_conv5x5 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 深度可分离卷积分支 (7x7)
        self.dw_conv7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=bias)
        self.pw_conv7x7 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 点卷积分支 (最终输出)
        self.pw_conv_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.dw_conv_dilated = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=bias)
        self.pw_conv_dilated = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        # 激活函数
        self.gelu = nn.GELU()

    def forward(self, x):
        # 3x3 深度可分离卷积分支
        branch_3x3 = self.dw_conv3x3(x)
        branch_3x3 = self.pw_conv3x3(branch_3x3)
        branch_3x3 = self.gelu(branch_3x3)

        # 5x5 点卷积分支 (通过1x1实现)
        branch_5x5 = self.pw_conv5x5(x)
        branch_5x5 = self.gelu(branch_5x5)

        # 7x7 深度可分离卷积分支
        branch_7x7 = self.dw_conv7x7(x)
        branch_7x7 = self.pw_conv7x7(branch_7x7)
        branch_7x7 = self.gelu(branch_7x7)
        branch_dilated = self.pw_conv_dilated(self.gelu(self.dw_conv_dilated(x)))
        # 特征融合
        out = branch_3x3 + branch_5x5 + branch_7x7 + branch_dilated

        # 通道注意力
        attention = self.channel_attention(out)
        out = out * attention

        out = self.pw_conv_out(out)
        return out


class DDGM(nn.Module):
    def __init__(self, in_channels, cond_dim=256, hidden_channels=None, reduction=16, dw=1, window_size=None,
                 norm='backward', bias=False):
        """
        Args:
            in_channels (int): 输入I的通道数
            cond_dim (int): 条件输入P的维度
            hidden_channels (int): 隐藏层通道数，如果不指定则使用in_channels
            reduction (int): 通道注意力的缩减比例
            dw (int): 频域变换的扩展因子
            window_size (int): 窗口大小，用于处理非固定尺寸输入
            norm (str): FFT的归一化方式
            bias (bool): 是否使用偏置
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels


        # 条件融合 - 增加复杂度
        self.kernel = nn.Sequential(
            nn.Linear(cond_dim, cond_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim // 2, in_channels * 2, bias=False),
        )
        # 添加特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=bias),
            nn.GELU()
        )
        # 归一化层
        self.norm1 = nn.LayerNorm(in_channels)

        # FCM分支 - 频域增强模块
        self.FCM = frequency_selection(
            dim=in_channels,
            dw=dw,
            norm=norm,
            act_method=nn.GELU,
            window_size=window_size,
            bias=bias
        )

        # MPU分支 - 多尺度处理单元
        self.mpu = MPU(dim=in_channels, bias=bias)

    def forward(self, P, I):
        """
        Args:
            P (torch.Tensor): 条件输入, 形状 (b, cond_dim)
            I (torch.Tensor): 图像输入, 形状 (b, c, h, w)

        Returns:
            torch.Tensor: 输出特征图, 形状 (b, c, h, w)
        """
        b, c, h, w = I.shape

        # 条件融合
        k_v = self.kernel(P).view(-1, c * 2, 1, 1)
        k_v1, k_v2 = k_v.chunk(2, dim=1)

        # 归一化并应用条件
        # 将I从(b,c,h,w)转换为(b,h*w,c)进行LayerNorm，然后转换回来
        I_norm = I.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)
        I_norm = self.norm1(I_norm)
        I_norm = I_norm.permute(0, 2, 1).view(b, c, h, w)  # (b, c, h, w)

        # 应用条件
        x = I_norm * k_v1 + k_v2

        # 双分支处理
        # FCM分支 - 频域增强模块
        FCM_out = self.FCM(x)

        # MPU分支 - 多尺度处理单元
        mpu_out = self.mpu(x)

        # 特征融合
        combined = torch.cat([FCM_out, mpu_out], dim=1)
        fused = self.feature_fusion(combined)

        # 残差连接
        output = I + fused
        return output



class HSSM(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, input_resolution):
        super(HSSM, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # 使用新的FeedForward类，设置hidden_channels为扩展后的维度
        self.ffn = DDGM(
            in_channels=dim,
            cond_dim=256,  # PFEM输出维度为256
            hidden_channels=int(dim * ffn_expansion_factor),
            reduction=8,
            dw=1,
            window_size=None,
            norm='ortho'
        )

        # 创建ResidualGroup实例
        self.residual_group = ESSM(
            dim=dim,
            input_resolution=input_resolution,
            depth=2,  # 每个HSSM包含6个ResidualGroup
            d_state=16,
            mlp_ratio=2.0,
            drop_path=0.1,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            img_size=input_resolution[0],  # 假设正方形输入
            patch_size=1,
            resi_connection='1conv',
            is_light_sr=False
        )

        # 用于序列和特征图转换
        self.patch_embed = PatchEmbed(
            img_size=input_resolution[0],
            patch_size=1,
            in_chans=dim,
            embed_dim=dim,
            norm_layer=None
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=input_resolution[0],
            patch_size=1,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None
        )
        self.kernel = nn.Sequential(
            nn.Linear(256, dim * 2, bias=False),
        )

    def forward(self, y):
        x, k_v = y
        B, C, H, W = x.shape

        # 保存原始的k_v，用于后续的FeedForward
        original_k_v = k_v  # 形状: (batch, 256)

        # 第一部分：ResidualGroup处理
        norm_x = self.norm1(x)
        b, c, h, w = x.shape

        # 为ResidualGroup处理生成条件权重
        k_v_processed = self.kernel(k_v).view(-1, c * 2, 1, 1)  # 形状: (batch, c*2, 1, 1)
        k_v1, k_v2 = k_v_processed.chunk(2, dim=1)
        conditioned_x = x * k_v1 + k_v2

        # 转换为序列 (B, C, H, W) -> (B, L, C)
        x_seq = norm_x.flatten(2).permute(0, 2, 1)
        # 通过ResidualGroup
        res_group_out = self.residual_group(x_seq, (H, W))
        # 转换回特征图 (B, L, C) -> (B, C, H, W)
        res_group_out = res_group_out.permute(0, 2, 1).view(B, C, H, W)
        x = conditioned_x + res_group_out

        # 第二部分：FFN处理
        norm2_x = self.norm2(x)
        # 使用原始的k_v（256维）传入FeedForward
        ffn_out = self.ffn(original_k_v, norm2_x)
        x = x + ffn_out

        # 返回时使用原始的k_v，保持维度一致性
        return [x, original_k_v]

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        avg_pool = nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, channels)
        max_pool = nn.functional.adaptive_max_pool2d(x, (1, 1)).view(batch_size, channels)
        avg_weight = self.fc(avg_pool)
        max_weight = self.fc(max_pool)
        weight = torch.sigmoid(avg_weight + max_weight).view(batch_size, channels, 1, 1)
        out = x * weight
        return out


class HSSMNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 scale=4,
                 dim=48,
                 num_blocks=[4, 5, 6, 7],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  # Other option 'BiasFree'
                 ):
        super(CDMSRNet1, self).__init__()
        self.scale = scale
        if self.scale == 2:
            inp_channels = 12
            self.pixel_unshuffle = nn.PixelUnshuffle(2)
        elif self.scale == 1:
            inp_channels = 48
            self.pixel_unshuffle = nn.PixelUnshuffle(4)
        else:
            inp_channels = 3

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 计算不同层级的输入分辨率
        self.initial_resolution = (64, 64)  # 假设输入为64x64
        res_level1 = self.initial_resolution
        res_level2 = (res_level1[0] // 2, res_level1[1] // 2)
        res_level3 = (res_level2[0] // 2, res_level2[1] // 2)
        res_level4 = (res_level3[0] // 2, res_level3[1] // 2)

        # 编码器部分
        self.encoder_level1 = nn.Sequential(*[
            HSSM(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                input_resolution=res_level1
            ) for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            HSSM(
                dim=int(dim * 2),
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                input_resolution=res_level2
            ) for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[
            HSSM(
                dim=int(dim * 4),
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                input_resolution=res_level3
            ) for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 4))
        self.latent = nn.Sequential(*[
            HSSM(
                dim=int(dim * 8),
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                input_resolution=res_level4
            ) for _ in range(num_blocks[3])
        ])

        # 解码器部分
        self.up4_3 = Upsample(int(dim * 8))
        self.ca4_3 = ChannelAttention(int(dim * 8))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            HSSM(
                dim=int(dim * 4),
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                input_resolution=res_level3
            ) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 4))
        self.ca3_2 = ChannelAttention(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            HSSM(
                dim=int(dim * 2),
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                input_resolution=res_level2
            ) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2))
        self.ca2_1 = ChannelAttention(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            HSSM(
                dim=int(dim * 2),
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                input_resolution=res_level1
            ) for _ in range(num_blocks[0])
        ])

        # 修复未解析的引用问题：移除多余的括号
        self.refinement = nn.Sequential(*[
            HSSM(
                dim=int(dim * 2),
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                input_resolution=res_level1
            ) for _ in range(num_refinement_blocks)
        ])

        # 尾部上采样
        modules_tail = [
            common.Upsampler(common.default_conv, 4, int(dim * 2)),
            nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, padding=1)
        ]
        self.tail = nn.Sequential(*modules_tail)
        self.pixelshuffle_inp = nn.PixelShuffle(self.scale)

    def forward(self, inp_img, k_v):
        if self.scale == 2:
            feat = self.pixel_unshuffle(inp_img)
        elif self.scale == 1:
            feat = self.pixel_unshuffle(inp_img)
        else:
            feat = inp_img

        inp_enc_level1 = self.patch_embed(feat)
        out_enc_level1, _ = self.encoder_level1([inp_enc_level1, k_v])  # (9,64,64,64)

        inp_enc_level2 = self.down1_2(out_enc_level1)  # (9,128,32,32)
        out_enc_level2, _ = self.encoder_level2([inp_enc_level2, k_v])

        inp_enc_level3 = self.down2_3(out_enc_level2)  # (9,256,16,16)
        out_enc_level3, _ = self.encoder_level3([inp_enc_level3, k_v])

        inp_enc_level4 = self.down3_4(out_enc_level3)  # (9,512,8,8)
        latent, _ = self.latent([inp_enc_level4, k_v])

        # 解码器部分
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.ca4_3(inp_dec_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _ = self.decoder_level3([inp_dec_level3, k_v])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.ca3_2(inp_dec_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _ = self.decoder_level2([inp_dec_level2, k_v])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.ca2_1(inp_dec_level1)
        out_dec_level1, _ = self.decoder_level1([inp_dec_level1, k_v])

        out_dec_level1, _ = self.refinement([out_dec_level1, k_v])
        out_dec_level1 = self.tail(out_dec_level1) + F.interpolate(inp_img, scale_factor=self.scale, mode='bicubic')
        # todo 把上采样图片的方式从near改为bicubic
        return out_dec_level1


class PFEM(nn.Module):
    def __init__(self, n_feats=64, n_encoder_res=6, scale=4):
        super(PFEM, self).__init__()
        self.scale = scale
        if scale == 2:
            E1 = [nn.Conv2d(60, n_feats, kernel_size=3, padding=1),
                  nn.LeakyReLU(0.1, True)]
        elif scale == 1:
            E1 = [nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
                  nn.LeakyReLU(0.1, True)]
        else:
            E1 = [nn.Conv2d(51, n_feats, kernel_size=3, padding=1),
                  nn.LeakyReLU(0.1, True)]
        E2 = [
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(*E)
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.pixel_unshufflev2 = nn.PixelUnshuffle(2)

    def forward(self, x, gt):
        gt0 = self.pixel_unshuffle(gt)
        if self.scale == 2:
            feat = self.pixel_unshufflev2(x)
        elif self.scale == 1:
            feat = self.pixel_unshuffle(x)
        else:
            feat = x
        x = torch.cat([feat, gt0], dim=1)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        S1_IRR = []
        fea1 = self.mlp(fea)
        S1_IRR.append(fea1)
        return fea1, S1_IRR


@ARCH_REGISTRY.register()
class CDMSRNet1(nn.Module):
    def __init__(self,
                 n_encoder_res=6,
                 inp_channels=3,
                 out_channels=3,
                 scale=4,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(CDMSRNet1, self).__init__()
        # Generator
        self.G = HSSMNet(
            inp_channels=inp_channels,
            out_channels=out_channels,
            scale=scale,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.E = PFEM(n_feats=64, n_encoder_res=n_encoder_res, scale=scale)


    def forward(self, x, gt):
        if self.training:
            IPFS1, S1_IPF = self.E(x, gt)  # todo 输入先经过PFEM，然后图像先验IPF和lr图片x再经过HSSMNet
            sr = self.G(x, IPFS1)
            return sr, S1_IPF  # todo 其中sr是超分结果，S1_IPF是一维图像先验
        else:
            IPFS1, _ = self.E(x, gt)
            sr = self.G(x, IPFS1)
            return sr