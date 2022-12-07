from ddpg_agent import ddpg
from melody import melody_learning

if __name__ == '__main__':
# setting
    Number_of_split = 40
    Learn_melody = False
    
#####################################################################
    is_sample = False
    sample = 0
    if Learn_melody:
        m_learning = melody_learning(n_split=Number_of_split)
        sample = m_learning.learning()
        is_sample = True
    player = ddpg(n_split=Number_of_split,is_sample=is_sample,sample=sample)
    player.learn()