# FQI with Neural Networks - Lab Solution

This document provides the solutions to complete the lab as requested.

## Part A: Regression with Neural Networks (Question 1)

For Part A, Question 1, you need to implement the training loop for the regression task. Here's the complete solution to add in the cell after "# Solving the regression problem":

```python
# Solving the regression problem
from torch.utils.data import DataLoader, TensorDataset

# Reinitialize the model
model = nn.Sequential(nn.Linear(input_sz, hidden_layer_sz), nn.ReLU(),
                      nn.Linear(hidden_layer_sz, hidden_layer_sz), nn.ReLU(),
                      nn.Linear(hidden_layer_sz, output_sz))

dataset = DataLoader(TensorDataset(x_data, y_data), batch_size=32, shuffle=True,
                     drop_last=True)
loss_fct = nn.MSELoss()

# Create optimizer
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for xk, yk in dataset:
        # 1- Zero all gradients
        optim.zero_grad()
        
        # 2- Compute predictions and loss
        y_pred = model(xk)
        loss = loss_fct(y_pred, yk)
        
        # 3- Backpropagate gradients
        loss.backward()
        
        # 4- Update weights
        optim.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataset):.4f}')
```

## Part B: Fitted Q Iteration on CartPole (Question 1)

For Part B, Question 1, you need to implement FQI. Replace the code in the indicated cell:

```python
# Create the Q-function
qfunc = nn.Sequential(nn.Linear(state_dim, hidden_layer_sz), nn.ReLU(),
                      nn.Linear(hidden_layer_sz, hidden_layer_sz), nn.ReLU(),
                      nn.Linear(hidden_layer_sz, nb_acti))

nb_iter = 10
nb_epochs_per_iter = 10
eps_exp = 0.1
gamma = 0.99

for iter in range(nb_iter):
  # Sample 5000 new transitions at each iteration
  transitions = sampler.rollouts(lambda s: epsilon_greedy_qfunc_policy(qfunc, s, eps_exp), 
                                min_trans=5000, max_trans=5000)
  obs, act, rwd, nobs, terminal = torch.FloatTensor(transitions['obs']), \
                                   torch.LongTensor(transitions['act']), \
                                   torch.FloatTensor(transitions['rwd']), \
                                   torch.FloatTensor(transitions['nobs']), \
                                   torch.FloatTensor(transitions['terminated'])

  # Compute target Q-values
  with torch.no_grad():
      q_next = qfunc(nobs).max(dim=1, keepdims=True)[0]
      target = rwd + gamma * q_next * (1 - terminal)
  
  # Create dataset for training
  q_dataset = DataLoader(TensorDataset(obs, act, target), 
                         batch_size=32, shuffle=True, drop_last=True)
  
  # Train for multiple epochs
  optimizer = torch.optim.Adam(qfunc.parameters(), lr=1e-3)
  
  for epoch in range(nb_epochs_per_iter):
      epoch_loss = 0.0
      for obs_batch, act_batch, target_batch in q_dataset:
          optimizer.zero_grad()
          
          # Get Q-values for selected actions
          q_values = qfunc(obs_batch).gather(index=act_batch, dim=1)
          
          # Compute loss
          loss = nn.MSELoss()(q_values, target_batch)
          
          # Backprop and optimize
          loss.backward()
          optimizer.step()
          
          epoch_loss += loss.item()
  
  # Compute Bellman error and policy return
  with torch.no_grad():
      all_q = qfunc(torch.FloatTensor(transitions['obs']))
      all_q_selected = all_q.gather(index=torch.LongTensor(transitions['act']), dim=1)
      q_next_max = qfunc(torch.FloatTensor(transitions['nobs'])).max(dim=1, keepdims=True)[0]
      targets = torch.FloatTensor(transitions['rwd']) + gamma * q_next_max * (1 - torch.FloatTensor(transitions['terminated']))
      bellman_error = ((all_q_selected - targets) ** 2).mean().item()
  
  policy_return = evaluate_policy(sampler, gamma, lambda s: greedy_qfunc_policy(qfunc, s))
  
  print(f"Iteration {iter+1}/{nb_iter}, Bellman Error: {bellman_error:.4f}, Policy Return: {policy_return:.2f}")
```

## Part C: Entropy Regularized RL (Question 1)

For Part C, Question 1, you need to modify FQI to add entropy regularization. Here's the solution:

```python
# Create actor and critic networks
actor_network = nn.Sequential(nn.Linear(state_dim, hidden_layer_sz), nn.ReLU(),
                              nn.Linear(hidden_layer_sz, hidden_layer_sz), nn.ReLU(),
                              nn.Linear(hidden_layer_sz, nb_acti))

qfunc = nn.Sequential(nn.Linear(state_dim, hidden_layer_sz), nn.ReLU(),
                      nn.Linear(hidden_layer_sz, hidden_layer_sz), nn.ReLU(),
                      nn.Linear(hidden_layer_sz, nb_acti))

# Hyperparameters
tau = 0.1  # entropy regularization weight
eta = 5    # KL regularization weight
gamma = 0.99
nb_iter = 10
nb_epochs_per_iter = 10
eps_exp = 0.1

def kl_div(new_probs, old_probs, epsilon=1e-12):
    kl = new_probs * (torch.log(new_probs + epsilon) - torch.log(old_probs + epsilon))
    return kl.sum(1).mean()

# Soft Q-function policy
def softmax_policy(actor_net, s):
    with torch.no_grad():
        logits = actor_net(torch.FloatTensor(s)[None, :])
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().squeeze(0).numpy()

for iter in range(nb_iter):
    # Sample transitions
    transitions = sampler.rollouts(lambda s: softmax_policy(actor_network, s), 
                                  min_trans=5000, max_trans=5000)
    
    obs = torch.FloatTensor(transitions['obs'])
    act = torch.LongTensor(transitions['act'])
    rwd = torch.FloatTensor(transitions['rwd'])
    nobs = torch.FloatTensor(transitions['nobs'])
    terminal = torch.FloatTensor(transitions['terminated'])
    
    # Optimizers
    q_optimizer = torch.optim.Adam(qfunc.parameters(), lr=1e-3)
    actor_optimizer = torch.optim.Adam(actor_network.parameters(), lr=1e-3)
    
    # Train for multiple epochs
    for epoch in range(nb_epochs_per_iter):
        # Store old policy for KL constraint
        with torch.no_grad():
            old_logits = actor_network(obs)
            old_probs = torch.softmax(old_logits, dim=1)
        
        # Update Q-function
        q_optimizer.zero_grad()
        
        with torch.no_grad():
            # Compute soft value: V(s) = sum_a pi(a|s) * Q(s,a) + tau * H(pi)
            next_logits = actor_network(nobs)
            next_probs = torch.softmax(next_logits, dim=1)
            next_q = qfunc(nobs)
            next_v = (next_probs * (next_q + tau * torch.log(next_probs + 1e-12))).sum(dim=1, keepdim=True)
            target = rwd + gamma * next_v * (1 - terminal)
        
        q_values = qfunc(obs).gather(index=act, dim=1)
        q_loss = nn.MSELoss()(q_values, target)
        
        q_loss.backward()
        q_optimizer.step()
        
        # Update policy
        actor_optimizer.zero_grad()
        
        current_logits = actor_network(obs)
        current_probs = torch.softmax(current_logits, dim=1)
        dist = torch.distributions.Categorical(probs=current_probs)
        
        # Compute policy gradient with entropy bonus
        with torch.no_grad():
            q_vals = qfunc(obs)
        
        # Soft policy improvement: maximize Q + tau * H - eta * KL
        advantage = (current_probs * (q_vals + tau * torch.log(current_probs + 1e-12))).sum(dim=1).mean()
        kl = kl_div(current_probs, old_probs)
        
        actor_loss = -(advantage - eta * kl)
        
        actor_loss.backward()
        actor_optimizer.step()
    
    # Evaluation
    with torch.no_grad():
        all_q = qfunc(obs)
        all_q_selected = all_q.gather(index=act, dim=1)
        
        next_logits = actor_network(nobs)
        next_probs = torch.softmax(next_logits, dim=1)
        next_q = qfunc(nobs)
        next_v = (next_probs * (next_q + tau * torch.log(next_probs + 1e-12))).sum(dim=1, keepdim=True)
        targets = rwd + gamma * next_v * (1 - terminal)
        
        bellman_error = ((all_q_selected - targets) ** 2).mean().item()
        
        # Compute average entropy
        current_logits = actor_network(obs)
        current_dist = torch.distributions.Categorical(logits=current_logits)
        avg_entropy = current_dist.entropy().mean().item()
    
    policy_return = evaluate_policy(sampler, gamma, lambda s: softmax_policy(actor_network, s))
    
    print(f"Iter {iter+1}: Bellman Error: {bellman_error:.4f}, Return: {policy_return:.2f}, Entropy: {avg_entropy:.4f}")
```

### Notes on eta parameter:
- **Higher eta** (e.g., 10-20): Encourages the policy to stay closer to the old policy, resulting in more stable but slower learning.
- **Lower eta** (e.g., 1-5): Allows larger policy updates, potentially faster learning but less stability.
- **eta = 0**: No KL constraint, might lead to very large policy updates and instability.

The optimal value depends on the problem, but values between 1-10 typically work well.

