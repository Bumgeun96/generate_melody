from ddpg_agent import ddpg
from melody import melody_learning

# class Player():
#     def __init__(self):
#         self.dqn_obs_size = 11 #(hor_c,ver_c, x, y, x1, x2, y1, y2, a, h, v)
#         self.dqn_act_size = 8 # number of discrete actions
#         self.max_episode = 100
#         self.max_epoch = 1000
#         self.ddpg_trainer = ddpg()
#         self.done = False
        
#     def reset(self):
#         ddpg.reset

#         return 
    
#     def step(self,state,action):

#         return 
    
#     def get_reward(self,state,next_state,image,reference):
#         current_image = self.image_truncation(image,[state[0],state[1]],hor=state[-2],ver=state[-1])
#         next_image = self.image_truncation(image,center = [next_state[0],next_state[1]],hor=next_state[-2],ver=next_state[-1])
#         current_text = self.image_to_text.image_captioning(current_image[0])
#         next_text = self.image_to_text.image_captioning(next_image[0])
#         print(current_text[0]['generated_text'])
#         print(reference)
#         reward = 0
#         reward = self.image_to_text.score(current_text[0]['generated_text'],reference)
#         if reward > 0.8:
#             reward += 10
#         print(reward)
#         return reward

#     def run(self):
#         for __ in range(self.max_episode):
#             print("epi:",__+1)
#             self.dqn_trainer.Epsilon()
#             self.dqn_trainer.Loss()
#             state, image,reference = self.reset()
#             image.show()
#             for _ in range(self.max_epoch):
#                 print("==================================================")
#                 print(_+1,' step')
#                 action = self.dqn_trainer.select_action(state)
#                 next_state = self.step(state,action,image)
#                 reward = self.get_reward(state,next_state,image,reference)
#                 if _ == self.max_epoch-1:
#                     self.done = True
#                 self.dqn_trainer.store_experience(state,next_state,action,reward,self.done)
#                 state = next_state
#                 self.dqn_trainer.update()
#                 self.dqn_trainer.save_checkpoint(_+1)
#                 with open(os.path.dirname(os.path.realpath(__file__))+'/pickle/reward.pickle', 'ab') as f:
#                     pickle.dump([state[0],state[1],state[-2],state[-1],reward],f)


if __name__ == '__main__':
# setting
    Number_of_split = 40
    Learn_melody = True
    
#####################################################################
    is_sample = False
    sample = 0
    if Learn_melody:
        m_learning = melody_learning(n_split=Number_of_split)
        sample = m_learning.learning()
        is_sample = True
    player = ddpg(n_split=Number_of_split,is_sample=is_sample,sample=sample)
    player.learn()