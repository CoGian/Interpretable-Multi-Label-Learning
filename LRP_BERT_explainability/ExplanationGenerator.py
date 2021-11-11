import numpy as np
import torch

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

class Generator:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def generate_LRP(self, input_ids, attention_mask, start_layer=11):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}

        indexes = [i for i, j in enumerate(torch.sigmoid(output).cpu().detach().numpy()[0]) if j >= .5]

        explanations = []

        for index in indexes:
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(self.device)
            one_hot = torch.sum(one_hot * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

            cams = []
            blocks = self.model.bert.encoder.layer
            for blk in blocks:
                grad = blk.attention.self.get_attn_gradients()
                cam = blk.attention.self.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            rollout[:, 0, 0] = 0
            explanation = rollout[:, 0][0]
            norm_explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min())
            explanations.append(norm_explanation)
        return explanations, output
