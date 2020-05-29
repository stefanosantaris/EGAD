# EGAD

This is a PyTorch implementation of the EGAD model as described in our submitted paper:

S. Antaris, D. Rafailidis, S. Girdzijauskas, EGAD: Evolving Graph Representation Learning with Self-Attention and Knowledge Distillation for Live Video Streaming Events

# Installation
```markdown
pip install -r requirements.txt
```

# Requirements
1. Python 3
2. NetworkX
3. scipy
4. PyTorch 1.5

# Run the demo
```markdown
python main.py
```


## Arguments
```markdown
start_graph:Starting graph
end_graph:Ending graph
num_exp:Number of experiments
teacher_embed_size:Teacher Embedding size
window:Window for evolution
teacher_n_heads:Number of Head Attention for the Teacher Model
dropout:Dropout
alpha:LeakyRelu alpha
learning_rate:Learning rate
student_emb:Student embedding
student_heads:Number of Head attention for the Student Model
distillation:Distillation Enabled
ns:Number of negative samples
gamma:Distillation balance
cuda:CUDA SUPPORT (0=FALSE/1=TRUE)

```

