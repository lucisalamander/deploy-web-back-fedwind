import torch
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Config
from einops import rearrange

from transformers import AutoModel
from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType


# --------- Conv + MLP Patch Embedding ---------
class ConvMLPPatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model, hidden_size=128, kernel_size=3):
        super().__init__()
        # Conv1d expects input shape [batch, in_channels, sequence_length]
        self.conv1d = nn.Conv1d(
            in_channels=patch_size,
            out_channels=patch_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.mlp = nn.Sequential(
            nn.Linear(patch_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model)
        )

    def forward(self, x):
        # x: [B*M, N, patch_size]  (N = số patch, patch_size = chiều patch)
        Bm, N, P = x.shape  # Bm = B * M
        # rearrange to [batch, in_channels, length] for Conv1d
        x = x.reshape(-1, P, N)
        v = self.conv1d(x)  # [B*M, patch_size, N]
        # permute so last dim = patch_size for MLP
        v = v.permute(0, 2, 1)  # [B*M, N, patch_size]
        out = self.mlp(v)  # [B*M, N, d_model]
        out = out.view(Bm, N, -1)  # [B*M, N, d_model]
        return out


# ---------- GPT4TS chính sửa ----------
class GPT4TS_Linear(nn.Module):
    def __init__(self, configs, device):
        super(GPT4TS_Linear, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2LMHeadModel(GPT2Config())
            self.gpt2.transformer.h = self.gpt2.transformer.h[:configs.llm_layers]
            print("gpt2 = {}".format(self.gpt2))

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        # --- Bước embedding mới ---
        outputs = self.in_layer(x)  # [B*M, N, d_model]

        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs


# ---------- GPT4TS Model ----------
class GPT4TS_Nonlinear(nn.Module):
    def __init__(self, configs, device):
        super(GPT4TS_Nonlinear, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        # compute number of patches
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        # initialize GPT2 backbone if needed
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2LMHeadModel.from_pretrained(
                    'gpt2',
                    output_attentions=True,
                    output_hidden_states=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
            else:
                self.gpt2 = GPT2LMHeadModel(GPT2Config())
            self.gpt2.transformer.h = self.gpt2.transformer.h[:configs.llm_layers]
            self.gpt2 = self._apply_lora(
                self.gpt2,
                r=getattr(configs, "lora_r", 16),
                alpha=getattr(configs, "lora_alpha", 32),
                dropout=getattr(configs, "lora_dropout", 0.1),
                # task_type=TaskType.CAUSAL_LM,
                target_modules=getattr(
                    configs,
                    "lora_target_modules",
                    ["c_attn", "c_fc", "c_proj"],  # GPT-2 modules
                ),
            )
            if getattr(configs, "freeze", False) and configs.pretrain and self.is_gpt:
                for name, param in self.gpt2.named_parameters():
                    if ("lora_" in name) or ("ln" in name) or ("wpe" in name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        # embedding: conv + mlp patch
        self.in_layer = ConvMLPPatchEmbedding(
            patch_size=configs.patch_size,
            d_model=configs.d_model,
            hidden_size=configs.hidden_size,
            kernel_size=configs.kernel_size
        )
        # final linear to map all patches to prediction length
        # self.out_layer = nn.Linear(configs.d_model * self.patch_num,
        #                            configs.pred_len)
        
        # self.proj_to_gpt = nn.Linear(configs.d_model, 768)
        
        self.proj_to_gpt = nn.Linear(configs.d_model, 768)
        self.head_hidden = 768
        self.out_layer = nn.Linear(self.head_hidden, configs.pred_len)


        # freeze GPT2 if requested
        if configs.freeze and configs.pretrain:
            for name, param in self.gpt2.named_parameters():
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # move modules to device
        # for module in (getattr(self, 'gpt2', None), self.in_layer, self.out_layer):
        for module in (getattr(self, 'gpt2', None), self.in_layer, self.proj_to_gpt, self.out_layer):
            if module is not None:
                module.to(device)
                module.train()

    def forward(self, x, itr):
        B, L, M = x.shape
        # normalization
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True,
                                     unbiased=False) + 1e-5).detach()
        x = x / stdev

        # rearrange to [B, M, L]
        x = rearrange(x, 'b l m -> b m l')
        # pad and unfold into patches
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1,
                     size=self.patch_size,
                     step=self.stride)
        # reshape to [B*M, N, patch_size]
        x = rearrange(x, 'b m n p -> (b m) n p')

        # conv+mlp embedding
        outputs = self.in_layer(x)  # [B*M, N, d_model]

        # pass through GPT2 if enabled
        if self.is_gpt:
    # project small d_model -> GPT hidden size
            outputs = self.proj_to_gpt(outputs)  # [B*M, N, 768]

    # run GPT transformer
            outputs = self.gpt2.transformer(inputs_embeds=outputs).last_hidden_state  # [B*M, N, 768]

    # pool across sequence (mean pooling)
            outputs = outputs.mean(dim=1)  # [B*M, 768]

    # map to prediction length
            outputs = self.out_layer(outputs)  # [B*M, pred_len]

        # reshape back to [B, pred_len, M]
            outputs = outputs.view(B, M, -1)
            outputs = rearrange(outputs, 'b m l -> b l m')
        else:
        # non-GPT path (original)
            outputs = outputs.reshape(B * M, -1)
            outputs = self.out_layer(outputs)
            outputs = rearrange(outputs, '(b m) l -> b l m', b=B)


        # denormalize
        outputs = outputs * stdev
        outputs = outputs + means
        return outputs

    def _apply_lora(self, model, r=16, alpha=32, dropout=0.1, target_modules=None):
        if target_modules is None:
            target_modules = ["c_attn", "c_fc", "c_proj"]
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        # Optional: show counts
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
        return model


class GPT4TS_Medium_Nonlinear(nn.Module):
    def __init__(self, configs, device):
        super(GPT4TS_Medium_Nonlinear, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if configs.is_gpt:
            self.gpt2 = AutoModel.from_pretrained(
                "gpt2-medium",
                output_hidden_states=True,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            # else:
            #     print("------------------no pretrain------------------")
            #     self.gpt2 = AutoModel.from_pretrained(
            #         "gpt2-medium",
            #         output_hidden_states=True,
            #         low_cpu_mem_usage=True,
            #         use_safetensors=True,
            #     )
            #     self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.transformer.h = self.gpt2.transformer.h[:configs.llm_layers]
            print("gpt2 = {}".format(self.gpt2))

        # --- THAY self.in_layer BẰNG MODULE MỚI ---
        self.in_layer = ConvMLPPatchEmbedding(
            patch_size=configs.patch_size,
            d_model=configs.d_model,
            hidden_size=configs.hidden_size,
            kernel_size=configs.kernel_size
        )
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        # --- Bước embedding mới ---
        outputs = self.in_layer(x)  # [B*M, N, d_model]

        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs

