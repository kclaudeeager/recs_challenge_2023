import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, InnerProductInteraction
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss


class MMoE_Layer(nn.Module):
    def __init__(self, num_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(MMoE_Layer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_experts)])
        self.gate = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                             hidden_units=gate_hidden_units,
                                             output_dim=num_experts,
                                             hidden_activations=hidden_activations,
                                             output_activation=None,
                                             dropout_rates=net_dropout,
                                             batch_norm=batch_norm) for _ in range(self.num_tasks)])
        self.gate_activation = get_activation('softmax')

    def forward(self, x):
        experts_output = torch.stack([self.experts[i](x) for i in range(self.num_experts)],
                                     dim=1)  # (?, num_experts, dim)
        mmoe_output = []
        for i in range(self.num_tasks):
            gate_output = self.gate[i](x)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)  # (?, num_experts)
            mmoe_output.append(torch.sum(torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1))
        return mmoe_output


class MMoE_SOFT(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="MMoE",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_experts=4,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 cvr_weight=1,
                 soft_weight=0.1,
                 soft_tau=0.1,
                 **kwargs):
        super(MMoE_SOFT, self).__init__(feature_map,
                                   task=task,
                                   num_tasks=num_tasks,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.loss_weight = nn.Parameter(torch.tensor([1., cvr_weight]) / (1+cvr_weight) * 2, requires_grad=False)
        self.soft_weight=soft_weight
        self.soft_tau = soft_tau
        
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.mmoe_layer = MMoE_Layer(num_experts=num_experts,
                                     num_tasks=self.num_tasks,
                                     input_dim=embedding_dim * feature_map.num_fields,
                                     expert_hidden_units=expert_hidden_units,
                                     gate_hidden_units=gate_hidden_units,
                                     hidden_activations=hidden_activations,
                                     net_dropout=net_dropout,
                                     batch_norm=batch_norm)
        self.tower = nn.ModuleList([MLP_Block(input_dim=expert_hidden_units[-1],
                                              output_dim=1,
                                              hidden_units=tower_hidden_units,
                                              hidden_activations=hidden_activations,
                                              output_activation=None,
                                              dropout_rates=net_dropout,
                                              batch_norm=batch_norm)
                                    for _ in range(num_tasks)])
        self.batch_norm = torch.nn.BatchNorm1d(num_features=128,affine=True)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.z_layers_0 = torch.nn.Linear(512, 256, bias=False,device='cuda:{}'.format(gpu))
        self.z_layers_1 = torch.nn.Linear(256, 128, bias=False,device='cuda:{}'.format(gpu))
        

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        expert_output = self.mmoe_layer(feature_emb.flatten(start_dim=1))
        tower_output = [self.tower[i](expert_output[i]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return_dict["expert_output"] = expert_output[1] 
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        labels = self.feature_map.labels
        loss = self.loss_fn[1](return_dict["{}_pred".format(labels[1])], y_true[1], reduction='mean')
        idx0 = torch.argwhere(y_true[0]==0)[:,0] # (?, )
        idx1 = torch.argwhere(y_true[0]==1)[:,0] # (?, )
        batch_sample = self.batch_norm(self.z_layers_1(self.z_layers_0(return_dict["expert_output"])))
        z_dict = {
            1:batch_sample[idx1,:],
            0:batch_sample[idx0,:]
        } # (?, 512)
        soft_loss = 0

        for pos_class in [0, 1]:
            z_pos = z_dict[pos_class]
            mm_mat = torch.mm(z_pos, batch_sample.T)
            inside = mm_mat/self.soft_tau
            c = torch.max(inside)
            right = torch.log(torch.sum(torch.exp(inside-c),dim=1,keepdim=True))+c
            left = torch.mm(z_pos, z_dict[pos_class].T)/self.soft_tau
            log_sum = left-torch.broadcast_to(right, left.shape)
            log_mean = torch.nanmean(log_sum,dim=1,keepdim=False)
            soft_loss-= torch.nanmean(log_mean)/2
        loss = loss + self.soft_weight*soft_loss
        return loss