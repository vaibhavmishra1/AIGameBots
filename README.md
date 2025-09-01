# AIGameBots


spatial and temporal attention are giving almost the same loss.
that means only the current instance of the main agent is main contribution for predicting the next step? 
need to verify this by using only the agent current features with an simple feed forward model. 
also test by joint temporal and spatial to see any decrease in loss. 


experiment - tiger = predict the next delta_x and delta_y of agent 
data =  20x10x6 
20 =  timesteps
10 - agents
6 - feature_dim = [ team_id, rel_x, rel_z, shr_key, deltax, deltay]
dataset  = "dataset_exp_tiger_0p02_0p3_100000.h5"
loss =  MSE 
epochs = 30 
temporal  only = loss = train = 0.000749 val = 0.001431
spatial only =  loss =  train = 0.001142 val = 0.001898
both temporal and spatial = loss = train = 0.000510 val = 0.001491
only main agent current features = loss = train = 0.001698  val = 0.001736

need to add long term prediction, for next 5 steps. 
experiment - hawk =  predict next 5 delta_x and delta_y of agent
data =  15 x 10 x 6
15 = timesteps
10 = agents
6 = feature_dim = [ team_id, rel_x, rel_z, shr_key, deltax, deltay]
output = 5 * [dx, dy]
data = dataset_exp_hawk_0p02_0p3_100000.h5
loss =  MSE
epochs = 20
temporal  only = loss = train =  val = 
spatial only =  loss =  train =  val = 
both temporal and spatial = loss = train =  val = 
only main agent current features = loss = train =   val = 

