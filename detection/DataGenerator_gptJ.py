import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from transformers import GPTJForCausalLM
from modelscope.utils.hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os

class PIDDerivativeLRScheduler:
    """
    Learning rate scheduler using only the second‐order derivative term of a PID controller.
    Adjusts lr only when the current loss decrease rate exceeds the previous decrease rate.
    new_lr = base_lr + kd * (rate_now - rate_prev)  if rate_now > rate_prev else keep last lr
    """
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, kd: float):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.kd = kd
        self.prev_prev_loss = None
        self.prev_loss = None
        self.last_lr = base_lr
        # initialize all param groups to base_lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = base_lr

    def step(self, current_loss: float):
        # On the first two calls we just record losses and keep lr unchanged
        if self.prev_loss is None:
            # first call
            self.prev_loss = current_loss
            return self.last_lr

        if self.prev_prev_loss is None:
            # second call
            self.prev_prev_loss = self.prev_loss
            self.prev_loss = current_loss
            return self.last_lr

        # Compute first-order rates
        rate_now = self.prev_loss - current_loss
        rate_prev = self.prev_prev_loss - self.prev_loss

        # print(self.prev_prev_loss," ",self.prev_loss," ",current_loss)
        # Compute second-order derivative (acceleration of loss decrease)
        accel = rate_now - rate_prev

        # If loss is decreasing faster than before, adjust; otherwise keep last lr
        # if accel > 0:
        #     new_lr = self.base_lr  + accel*self.kd
        #     new_lr = max(new_lr, 1e-8)
        # else:
        #     new_lr = self.last_lr

        new_lr = self.base_lr  + accel*self.kd
        new_lr = max(new_lr, 1e-8)

        # Apply to all parameter groups
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr

        # Shift losses for next step
        self.prev_prev_loss = self.prev_loss
        self.prev_loss = current_loss
        self.last_lr = new_lr

        return new_lr


class InputOptimizer:
    def __init__(self,
                 model_name: str = 'AI-ModelScope/gpt-j-6b',
                 device: str = 'cuda',
                 batch_size: int = 1,
                 seq_len: int = 16,
                 hidden_size: int = 4096,
                 lr: float = 1e-2,
                 kd: float = 0.0):  # derivative gain
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.lr = lr
        self.kd = kd

        # Load and freeze model in FP32
        model_dir = snapshot_download(repo_id=model_name, cache_dir='/hy-tmp/sdpcos_2025/code/gpt-j-6b')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            weights_only=False
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.vocab_size = self.model.config.vocab_size
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def generate_target(self) -> torch.Tensor:
        """Generate random target probability distribution shape=(B, L, V)"""
        logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size, device=self.device)
        target = torch.softmax(logits, dim=-1)
        return target.clamp(min=1e-6)

    def init_input(self) -> torch.Tensor:
        """Generate random input embeddings shape=(B, L, H)"""
        return torch.randn(self.batch_size, self.seq_len, self.hidden_size,
                           device=self.device, requires_grad=True)

    def optimize(self,
                 num_steps: int = 100,
                 print_every: int = 10) -> torch.Tensor:
        """
        Run backward propagation, optimize input embeddings,
        and adjust learning rate via PID derivative control.
        """
        target = self.generate_target()
        input_embeds = self.init_input()

        optimizer = torch.optim.Adam([input_embeds], lr=self.lr)
        # Initialize derivative-only PID scheduler
        scheduler = PIDDerivativeLRScheduler(optimizer, base_lr=self.lr, kd=self.kd)

        prev_loss = None
        rise_count = 0  # 记录连续上升次数

        for step in range(1, num_steps + 1):
            optimizer.zero_grad()
            outputs = self.model(inputs_embeds=input_embeds)
            log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
            loss = self.loss_fn(log_probs, target)
            loss.backward()
            # Gradient clipping
            clip_grad_norm_([input_embeds], max_norm=1.0)

            # 当前 loss
            current_loss = loss.item()

            # 调整学习率
            new_lr = scheduler.step(current_loss)
            optimizer.step()

            # 检查 loss 是否上升
            if prev_loss is not None and current_loss > prev_loss:
                rise_count += 1
            else:
                rise_count = 0
            prev_loss = current_loss

            # 如果连续 5 次上升，提前停止
            if rise_count >= 5:
                print(f'Early stopping at step {step}: loss increased for {rise_count} consecutive steps.')
                break

            if step % print_every == 0:
                print(f'[Step {step:4d}/{num_steps}]  Loss = {current_loss:.4f}  LR = {new_lr}')
        return input_embeds,target

    def save_data(self, 
                  input_embeds: torch.Tensor, 
                  target: torch.Tensor,
                  filepath: str = "./GPTJ_inputbatch.pkl",
                  include_metadata: bool = True):
        """
        Save input_embeds and target to a pickle file
        
        Args:
            input_embeds: Optimized input embeddings [B, L, H]
            target: Target probability distribution [B, L, V]
            filepath: Path to save the pickle file
            include_metadata: Whether to include metadata about the generation
        """
        # 创建保存目录
        save_dir = os.path.dirname(filepath)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 准备保存的数据
        data_dict = {
            'input_embeds': input_embeds.cpu(),  # 移到CPU以节省内存
            'target': target.cpu(),
            'shapes': {
                'input_embeds': input_embeds.shape,
                'target': target.shape
            }
        }
        
        # 添加元数据
        if include_metadata:
            data_dict['metadata'] = {
                'batch_size': self.batch_size,
                'seq_len': self.seq_len,
                'hidden_size': self.hidden_size,
                'vocab_size': self.vocab_size,
                'device': self.device,
                'lr': self.lr,
                'kd': self.kd,
                'data_types': {
                    'input_embeds': str(input_embeds.dtype),
                    'target': str(target.dtype)
                }
            }
        
        # 保存到pickle文件
        import pickle
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"数据已成功保存到: {filepath}")
            
            # 打印保存信息
            file_size = os.path.getsize(filepath)
            print(f"文件大小: {file_size / (1024*1024):.2f} MB")
            print(f"Input embeddings 形状: {input_embeds.shape}")
            print(f"Target 形状: {target.shape}")
            
        except Exception as e:
            print(f"保存失败: {e}")
            raise


if __name__ == "__main__":
    optimizer = InputOptimizer(
        model_name='AI-ModelScope/gpt-j-6b',
        device='cuda:0',
        batch_size=4,
        seq_len=32,
        hidden_size=4096,
        lr=1.2e-3,
        kd=1e-3  # example derivative gain
    )

    optimized_input,target = optimizer.optimize(num_steps=50000, print_every=20)
    optimizer.save_data(optimized_input,target)
    # optimized_input.shape == (4, 32, 4096)

