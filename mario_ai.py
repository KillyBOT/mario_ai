"""
Help mainly from: 
https://chrispresso.io/AI_Learns_To_Play_SMB_Using_GA_And_NN
https://github.com/Chrispresso/SuperMarioBros-AI/blob/master/utils.py
https://www.youtube.com/watch?v=DOAPB7xh1Ik&list=PLTWFMbPFsvz3CeozHfeuJIXWAJMkPtAdS&index=6&ab_channel=LucasThompson
https://github.com/vivek3141/super-mario-neat/blob/master/src/run.py
Credit them with pretty much everything, especially the boiler plate
code for finding enemies and whatever and the fitness function.
"""

from mario_ai_game_funcs import *
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import *
import gym_super_mario_bros
import retro
import numpy as np
import time
import sys, os
import neat
import cv2
import visualize
import pickle

gEnv = 0

def get_output_blocks(nn,ram):
	inputArray = []

	#ram = env.get_ram()
	tiles = get_tiles(ram)
	marioPos = get_mario_row_col(ram)
	marioPos = (min(marioPos[POS_X],9),marioPos[POS_Y])
	
	#Don't look at all the tiles, just the ones in front of Mario

	for x in range(marioPos[POS_X],marioPos[POS_X]+7):
		for y in range(4,RESOLUTION_Y//SPRITE_HEIGHT-1):
			inputArray.append(tiles[(y,x)])

	#Also add seven inputs for a one-hot encoding for mario's position
	for y in range(4, RESOLUTION_Y//SPRITE_HEIGHT-1):
		if marioPos[POS_Y] == y:
			inputArray.append(1)
		else:
			inputArray.append(0)

	nnOutput = nn.activate(inputArray)
			
	#IDK why the NN is being a stupid
	for p in range(len(nnOutput)):
		if nnOutput[p] > NODE_CUTOFF:
			nnOutput[p] = 1

	return nnOutput.index(max(nnOutput))

def get_output_raw(nn,env,obs,inx,iny):

	obs = cv2.resize(obs, (inx,iny))
	obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
	obs = np.reshape(obs,(inx,iny))

	inputArray = np.ndarray.flatten(obs)
	inputArray = np.interp(inputArray, (0,254), (-1,+1))

	nnOutput = nn.activate(inputArray)

	return nnOutput.index(max(nnOutput))
	#return nnOutput

class evaluator(object):
	def __init__(self, genome, config):
		self.genome = genome
		self.config = config

	def work(self):

		self.env = gym_super_mario_bros.make(SMB_VERSION)
		self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
		obs = self.env.reset()
		#nn = neat.nn.recurrent.RecurrentNetwork.create(self.genome,self.config)
		nn = neat.nn.FeedForwardNetwork.create(self.genome,self.config)

		iny,inx,inc = self.env.observation_space.shape
		inx //= SQUISH_FACTOR
		iny //= SQUISH_FACTOR

		frames = 0
		fitness = 0
		maxFitness = 0
		timeoutCounter = 0
		xPos = 0
		xPosMax = 0

		done = False

		while not done:
			frames += 1

			#self.env.render()

			#obs = cv2.resize(obs, (inx,iny))
			#obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
			#obs = np.reshape(obs, (inx,iny))

			#inputArray = np.ndarray.flatten(obs)
			#inputArray = np.interp(inputArray, (0,254), (-1,+1))

			#Look at what the tiles are
			nnOutput = get_output_blocks(nn,self.env.ram)
			#nnOutput = get_output_raw(nn,self.env,obs,inx,iny)

			obs, rew, done, info = self.env.step(nnOutput)

			#xPos = info[FITNESS_DETERMINER]
			fitness += rew

			if fitness > maxFitness:
				maxFitness = fitness
				timeoutCounter = 0
			else:
				timeoutCounter += 1

			if done or timeoutCounter >= TIMEOUT_TIME or info['life'] <= 1:
				done = True
			elif info['flag_get']:
				fitness += 1000000
				done = True
				#fitness = calc_fitness_2(info['xscrollLo'],info['time'])
				#fitness = calc_fitness(frames,xPos)
				#print("Genome: " + str(genomeID) + "\tFitness: " + str(genome.fitness))

			"""if timeoutCounter >= TIMEOUT_TIME:
				done = True
				fitness = calc_fitness(frames,xPos)"""




		#print("Fitness: " + str(fitness))

		self.env.close()
		return max(0.0,float(fitness))

def eval_genomes_parallel(genome, config):

	evaluation = evaluator(genome, config)
	return evaluation.work()

def eval_genomes_single(genomes,config):

	global gEnv

	for genomeID, genome in genomes:
		obs = gEnv.reset()
		actions = gEnv.action_space.sample()
		#nn = neat.nn.recurrent.RecurrentNetwork.create(genome,config)
		nn = neat.nn.FeedForwardNetwork.create(genome,config)
		iny,inx,inc = gEnv.observation_space.shape
		
		inx //= SQUISH_FACTOR
		iny //= SQUISH_FACTOR

		frame = 0
		timeoutCounter = 0
		fitness = 0
		maxFitness = 0

		done = False

		while not done:

			gEnv.render()
			frame += 1

			#print_tiles_in_front(gEnv.get_ram())
			nnOutput = get_output_blocks(nn,gEnv.ram)
			#nnOutput = get_output_raw(nn,gEnv,obs,inx,iny)
			#print(nnOutput)

			obs, rew, done, info = gEnv.step(nnOutput)

			fitness += rew

			if fitness > maxFitness:
				maxFitness = fitness
				timeoutCounter = 0
			else:
				timeoutCounter += 1

			if done or timeoutCounter >= TIMEOUT_TIME or info['life'] <= 1:
				done = True
				genome.fitness = max(0.0,float(fitness))
			elif info['flag_get']:
				genome.fitness = 1000000 + max(0.0,float(fitness))
				done = True
				#print("Fitness: " + str(genome.fitness))

			"""if timeoutCounter >= TIMEOUT_TIME:
				done = True
				genome.fitness = calc_fitness(frame,xPos)"""

def main_single(config):

	global gEnv

	gEnv = gym_super_mario_bros.make(SMB_VERSION)
	gEnv = JoypadSpace(gEnv, COMPLEX_MOVEMENT)

	population = neat.Population(config)

	if len(sys.argv) > 1:
		population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-'+str(sys.argv[1]))

	population.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	population.add_reporter(stats)
	population.add_reporter(neat.Checkpointer(CHECKPOINT_DIST))

	if len(sys.argv) > 1:
		best = population.run(eval_genomes_single,NUM_OF_GENERATIONS-int(sys.argv[1]))
	else:
		best = population.run(eval_genomes_single,NUM_OF_GENERATIONS)

	gEnv.close()

	pickle.dump(population.best_genome,open('best.pkl','wb'))
	visualize.draw_net(config,best,True)
	visualize.plot_stats(stats,ylog=False,view=True)
	visualize.plot_species(stats,view=True)

	render_best(best,config)

def main_parallel(config):

	population = neat.Population(config)

	if len(sys.argv) > 1:
		population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-'+str(sys.argv[1]))

	population.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	population.add_reporter(stats)
	population.add_reporter(neat.Checkpointer(CHECKPOINT_DIST))

	pe = neat.ParallelEvaluator(PARALLEL_NUM,eval_genomes_parallel)

	if len(sys.argv) > 1:
		best = population.run(pe.evaluate,NUM_OF_GENERATIONS-int(sys.argv[1]))
	else:
		best = population.run(pe.evaluate,NUM_OF_GENERATIONS)
	
	pickle.dump(best,open('best.pkl','wb'))
	pickle.dump(population.best_genome,open('bestestBest.pkl','wb'))
	visualize.draw_net(config,best,True)
	visualize.plot_stats(stats,ylog=False,view=True)
	visualize.plot_species(stats,view=True)

	#Render the best genome in a .bk2 file
	
	render_best(best,config)

def render_best(best,config):
	print("Rendering...")
	bestNN = neat.nn.FeedForwardNetwork.create(self.genome,self.config)

	bestEnv = gym_super_mario_bros.make(SMB_VERSION,record='.')
	bestEnv = JoypadSpace(bestEnv, COMPLEX_MOVEMENT)
	obs = bestEnv.reset()

	iny,inx,inc = bestEnv.observation_space.shape
	inx //= SQUISH_FACTOR
	iny //= SQUISH_FACTOR

	done = False

	while not done:
		bestEnv.render()
		#print_tiles(get_tiles(bestEnv.get_ram()))
		#nnOutput = get_output_blocks(bestNN, bestEnv)
		nnOutput = get_output_raw(bestNN, bestEnv, obs, inx,iny)
		
		obs, rew, done, info = bestEnv.step(nnOutput)
		if done:
			break

		#time.sleep(0.016)

	bestEnv.close()

if __name__ == "__main__":

	#['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

	config = neat.Config(
		neat.DefaultGenome,
		neat.DefaultReproduction,
		neat.DefaultSpeciesSet,
		neat.DefaultStagnation,
		CONFIG_FILE)

	main_parallel(config)
	#main_single(config)
	
	"""
	#env = retro.make('SuperMarioBros-Nes',LEVEL)
	env = gym_super_mario_bros.make(SMB_VERSION)
	env= JoypadSpace(env,COMPLEX_MOVEMENT)
	obs = env.reset()
	
	running = Trueself
	
	while running:

		action = env.action_space.sample()
		#action = [0,0,0,0,0,0,0,1,0] # Move forward
		#ram = env.get_ram()
		obs, rew, done, info = env.step(action)

		os.system("clear")
		print(info,rew)
		#sys.stdout.write('\x1b[1A\x1b[2K')		

		#tiles = get_tiles(ram)
		#for y in range(RESOLUTION_Y//SPRITE_HEIGHT):
		#	for x in range(RESOLUTION_X//SPRITE_WIDTH-1):
		#		print(tiles[(x,y)],end=" ")
		#	print("")

		if done:
			obs = env.reset()
		env.render()

		time.sleep(0.01)

	env.close()
	"""
	

	
	