# AIGameBots


spatial and temporal attention are giving almost the same loss.
that means only the current instance of the main agent is main contribution for predicting the next step? 
need to verify this by using only the agent current features with an simple feed forward model. 
also test by joint temporal and spatial to see any decrease in loss. 


experiment - tiger = predict the next delta_x and delta_y of agent 
dataset  = 
loss =  MSE 
temporal  only = 
spatial only =  loss =  
both temporal ans spatial = loss =
only main agent current features = loss = 


need to add long term prediction, for next 5 steps. 
