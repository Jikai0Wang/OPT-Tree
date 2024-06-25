import torch

class Tree:
    def __init__(self, nnodes, device, threshold,max_depth):
        self.nnodes=nnodes
        self.device=device
        self.threshold=threshold
        self.depth=0
        self.weight=0
        self.max_depth=max_depth
        self.weight_matrix=torch.zeros([max_depth,self.nnodes], device=self.device)
        self.input_ids_matrix=torch.zeros([max_depth,self.nnodes], dtype=torch.long,device=self.device)
        self.parents_matrix=torch.zeros([max_depth,self.nnodes], dtype=torch.long,device=self.device)
        self.kv_mask=torch.zeros([self.nnodes,0],dtype=torch.int8,device=self.device)
        self.tri=torch.eye(self.nnodes, dtype=torch.int8, device=self.device)
        self.rows = torch.arange(self.nnodes,device=self.device)
        self.position_id=torch.zeros([self.nnodes],dtype=torch.long,device=self.device)
        self.kv_cache_mask=torch.zeros([self.nnodes,self.nnodes],dtype=torch.int8,device=self.device)

    def initialize(self,logits):
        logits, ids = torch.topk(logits[0][-1], k=self.nnodes,dim=-1)
        self.weight_matrix[self.depth].copy_(logits)
        self.input_ids_matrix[self.depth].copy_(ids)
        self.parents_matrix[0].copy_(self.rows)
        output_dict={"input_ids":ids,
                     "position_ids":self.position_id+1,
                     "attention_mask":self.tri,
                     "parent_last":self.rows,
                     "is_final":False
                     }
        self.depth+=1
        return output_dict
    
    def add(self,logits):
        logits, ids = torch.topk(logits[0], k=self.nnodes,dim=-1)
        last_layer_weights=self.weight_matrix[self.depth-1].unsqueeze(1)
        logits=logits*last_layer_weights
        flat_logits,flat_ids = logits.view(-1),ids.view(-1)
        global_top_logits,global_top_idx=torch.topk(flat_logits,k=self.nnodes,dim=-1)
        input_ids=flat_ids[global_top_idx]
        parents=global_top_idx//self.nnodes
        self.parents_matrix[self.depth].copy_(parents)
        self.weight_matrix[self.depth].copy_(global_top_logits)
        self.input_ids_matrix[self.depth].copy_(input_ids)
        self.kv_cache_mask[self.rows, parents]=1
        self.kv_mask=torch.cat([self.kv_mask[parents],self.kv_cache_mask],dim=1)
        self.kv_cache_mask[self.rows, parents]=1
        attention_mask=torch.cat([self.kv_mask,self.tri],dim=1)
        output_dict={"input_ids":input_ids,
                     "position_ids":self.position_id+(self.depth+1),
                     "attention_mask":attention_mask,
                     "parent_last":parents,
                     "is_final":False
                     }
        self.depth+=1
        return output_dict
    
    def generate_attention_mask(self,parents):
        attention_mask = self.tri
        grandp=parents.clone()
        for _ in range(self.depth-2):
            attention_mask[self.rows,grandp]=1
            grandp=grandp[parents]
        return attention_mask

    def update(self,logits):
        if self.depth==0:
            return self.initialize(logits)
        outputs=self.add(logits)
        top_weights,top_index=torch.topk(self.weight_matrix[:self.depth-1].view(-1),k=self.nnodes,dim=-1)
        weight=top_weights.sum()
        if weight-self.weight>self.threshold and self.depth<self.max_depth:
            self.weight=weight
        else:
            rows = top_index // self.nnodes
            cols = top_index % self.nnodes
            sorted_indices=torch.argsort(rows)
            rows=rows[sorted_indices]
            cols=cols[sorted_indices]
            index=torch.arange(self.nnodes*(self.depth-1),device=self.device).view(-1,self.nnodes)
            input_ids=self.input_ids_matrix[rows,cols]
            position_ids=rows+1
            parents=self.parents_matrix[rows,cols]
            source_idx=index[rows,cols]
            parent_idx=index[torch.clamp_min(rows-1,0),parents]
            parents_index= torch.nonzero(source_idx.unsqueeze(0) == parent_idx.unsqueeze(1))[:,1]
            attention_mask=self.generate_attention_mask(parents_index)
            outputs={"input_ids":input_ids,
                     "position_ids":position_ids,
                     "attention_mask":attention_mask,
                     "parent_last":parents_index,
                     "is_final":True
                     }
        return outputs