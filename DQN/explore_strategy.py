import torch

class Agent():
    def __init__(self, env):
        self.env = env
        
        self.thompson_alpha = torch.ones(self.env.action_space.n)
        self.thompson_beta = torch.ones(self.env.action_space.n)
        
        self.action_values = torch.zeros(self.env.action_space.n) 
        self.action_values += 1/3 
        self.temperature = 0.1

    def get_boltzmann_action(self, action_values):
        
        exp_probabilities = torch.exp(action_values / self.temperature )
        probabilities = exp_probabilities / torch.sum(exp_probabilities)
        
        probabilities[-1] = 1 - torch.sum(probabilities[:-1])
        
        action = torch.multinomial(probabilities, num_samples=1).item()
        return action 

    def update_thompson_sampling_action(self, reward, action):
        self.thompson_alpha[action] += reward
        
        if self.thompson_alpha[action] <= 0 :
            self.thompson_alpha[action] = 1e-10 
        
        self.thompson_beta[action] += 1 - reward
        if self.thompson_beta[action] <= 0 :
            self.thompson_beta[action] = 1e-10    
            
            
    def get_thompson_sampling_action(self):
        
        theta = torch.distributions.Beta(self.thompson_alpha, self.thompson_beta).sample()
        return theta.argmax().item()
    
    def choose_action(self, mode, action_values):
        """
        Choose which action to take, based on the observation.
        If observation is seen for the first time, initialize its Q values to 0.0
        """
        action = -1

        if mode == 'thompson':
            action = self.get_thompson_sampling_action()
                
        if mode == 'boltzmann':
            action = self.get_boltzmann_action(action_values)
                    
        return action


