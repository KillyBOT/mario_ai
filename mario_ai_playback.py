import retro
import sys, os
import time
from mario_ai_game_funcs import *
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import *
import gym_super_mario_bros
import pickle
import neat
"""from mario_ai_game_funcs import *
import neat
import pickle"""

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


if __name__ == "__main__":
	if len(sys.argv) > 1:

		outputType = sys.argv[1].split(".")[1]
		
		if outputType == "blk":
			movie = retro.Movie(sys.argv[1])
			movie.step()

			env = retro.make(
				game=movie.get_game(),
				state = None,
				use_restricted_actions=retro.Actions.ALL,
				players=movie.players,
			)

			env.initial_state = movie.get_state()
			env.reset()

			while movie.step():
				keys = []
				for p in range(movie.players):
					for i in range(env.num_buttons):
						keys.append(movie.get_key(i,p))
				env.step(keys)
				env.render()
				time.sleep(0.016)

			env.close()

		elif outputType == "pkl":

			config = neat.Config(
				neat.DefaultGenome,
				neat.DefaultReproduction,
				neat.DefaultSpeciesSet,
				neat.DefaultStagnation,
			CONFIG_FILE)

			env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
			env = JoypadSpace(env,COMPLEX_MOVEMENT)
			obs = env.reset()
			genome = 0

			fitness = 0
			maxFitness = 0
			timeoutCounter = 0

			with open(sys.argv[1],'rb') as inputFile:
				genome = pickle.load(inputFile)

			nn = neat.nn.FeedForwardNetwork.create(genome, config)

			done = False

			while not done:

				env.render()

				nnOutput = get_output_blocks(nn,env.ram)

				obs, rew, done, info = env.step(nnOutput)

				fitness += rew

				if fitness > maxFitness:
					maxFitness = fitness
					timeoutCounter = 0
				else:
					timeoutCounter += 1

				if done or timeoutCounter >= TIMEOUT_TIME or info['life'] <= 1 or info['flag_get']:
					done = True

				time.sleep(0.01)

			env.close()

