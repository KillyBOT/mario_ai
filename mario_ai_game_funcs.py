import numpy as np
import sys, os

#Constants

FITNESS_DETERMINER = 'xscrollLo'
TIMEOUT_TIME = 60
NUM_OF_GENERATIONS = 2500
CHECKPOINT_DIST = 50
TIMEOUT_BONUS = 100
PARALLEL_NUM = 4
NODE_CUTOFF = 0.5
SQUISH_FACTOR = 8
SMB_VERSION = 'SuperMarioBros-v0'
CONFIG_FILE = 'config-smb-test'

#Locations in memory and other constants

POS_X = 0
POS_Y = 1

MAX_NUM_ENEMIES = 5
PAGE_SIZE = 256
NUM_BLOCKS = 8
RESOLUTION_X = 256
RESOLUTION_Y = 240
SPRITE_WIDTH = 16
SPRITE_HEIGHT = 16
STATUS_BAR_WIDTH = RESOLUTION_X
STATUS_BAR_HEIGHT = 2*SPRITE_HEIGHT
NUM_TILES = 416
NUM_SCREEN_PAGES = 2
TOTAL_RAM = NUM_BLOCKS * PAGE_SIZE

ENEMY_GREEN_KOOPA_1 = 0x00
ENEMY_RED_KOOPA1_   = 0x01
ENEMY_BUZZY_BEETLE = 0x02
ENEMY_RED_KOOPA_2 = 0x03
ENEMY_GREEN_KOOPA_2 = 0x04
ENEMY_HAMMER_BROTHER = 0x05
ENEMY_GOOMBA = 0x06
ENEMY_BLOOPER = 0x07
ENEMY_BULLET_BILL = 0x08
ENEMY_GREEN_KOOPA_PARATROOPA = 0x09
ENEMY_GREY_CHEEP_CHEEP = 0x0A
ENEMY_RED_CHEEP_CHEEP = 0x0B
ENEMY_POBODOO = 0x0C
ENEMY_PIRANHA_PLANT = 0x0D
ENEMY_GREEN_PARATROOPA_JUMP = 0x0E
ENEMY_BOWSER_FLAME1 = 0x10
ENEMY_LAKITU = 0x11
ENEMY_SPINY_EGG = 0x12
ENEMY_FLY_CHEEP_CHEEP = 0x14
ENEMY_BOWSER_FLAME2 = 0x15
ENEMY_GENERIC = 0xFF

TILE_EMPTY = 0x00
TILE_FAKE = 0x01
TILE_GROUND = 0x54
TILE_TOP_PIPE1 = 0x12
TILE_TOP_PIPE2 = 0x13
TILE_BOTTOM_PIPE_1 = 0x14
TILE_BOTTOM_PIPE_2 = 0x15
TILE_FLAGPOLE_TOP =  0x24
TILE_FLAGPOLE = 0x25
TILE_COIN_BLOCK_1 = 0xC0
TILE_COIN_BLOCK_2 = 0xC1 
TILE_COIN = 0xC2
TILE_BREAKABLE_BLOCK = 0x51
TILE_GENERIC = 0xFF

DYNAMIC_MARIO = 0xAA
DYNAMIC_STATIC_LIFT1 = 0x24
DYNAMIC_STATIC_LIFT2 = 0x25
DYNAMIC_VERTICAL_LIFT1 = 0x26
DYNAMIC_VERTICAL_LIFT2 = 0x27
DYNAMIC_HORIZONTAL_LIFT = 0x28
DYNAMIC_FALLING_STATIC_LIFT = 0x29
DYNAMIC_HORIZONTAL_MOVING_LIFT=  0x2A
DYNAMIC_LIFT1 = 0x2B
DYNAMIC_LIFT2 = 0x2C
DYNAMIC_VINE = 0x2F
DYNAMIC_FLAGPOLE = 0x30
DYNAMIC_START_FLAG = 0x31
DYNAMIC_JUMP_SPRING = 0x32
DYNAMIC_WARPZONE = 0x34
DYNAMIC_SPRING1 = 0x67
DYNAMIC_SPRING2 = 0x68
DYNAMIC_GENERIC = 0xFF

ENEMY_DRAWN = 0x0F
ENEMY_TYPE = 0x16
ENEMY_POS_X_LEVEL = 0x6E
ENEMY_POS_X_SCREEN = 0x87
ENEMY_POS_Y_SCREEN = 0xCF
PLAYER_POS_X_LEVEL = 0x06D
PLAYER_POS_X_SCREEN = 0x086
PLAYER_POS_X_SCREEN_OFFSET = 0x3AD
PLAYER_POS_Y_SCREEN_OFFSET = 0x3B8
ENEMY_XITION_SCREEN_OFFSET = 0x3AE
PLAYER_POS_Y_SCREEN = 0xCE
PLAYER_POSITION_VERTICAL_SCREEN = 0xB5

#Most of these functions I just took from the link above

def get_enemy_tile_pos(ram):
	enemies = list()

	for enemyNum in range(MAX_NUM_ENEMIES):
		enemy = ram[ENEMY_DRAWN + enemyNum]

		if enemy:
			#We need both the level and screen x position to be accurate
			xPosLevel = ram[ENEMY_POS_X_LEVEL+enemyNum]
			xPosScreen = ram[ENEMY_POS_X_SCREEN+enemyNum]

			xPos = (xPosLevel * RESOLUTION_X) + xPosScreen
			yPos = ram[ENEMY_POS_Y_SCREEN+enemyNum]

			enemyID = ram[ENEMY_TYPE+enemyNum]

			enemies.append((xPos,yPos,enemyID))

	return enemies

def get_score(ram):
	score = 0
	m = 10
	for p in range(0x07DC,0x07D7-1,-1):
		score += ram[p]*m
		m *= 10

	return score

def get_mario_pos_level(ram):
	x = ram[PLAYER_POS_X_LEVEL] * RESOLUTION_X + ram[PLAYER_POS_X_SCREEN]
	y = ram[PLAYER_POS_Y_SCREEN_OFFSET]
	return (x,y)

def get_mario_pos_screen(ram):
	x = ram[PLAYER_POS_X_SCREEN_OFFSET]
	y = ram[PLAYER_POS_Y_SCREEN] * ram[PLAYER_POSITION_VERTICAL_SCREEN] + SPRITE_HEIGHT

	return (x,y)

def get_mario_row_col(ram):
	x,y = get_mario_pos_screen(ram)

	y = ram[PLAYER_POS_Y_SCREEN_OFFSET] + 16
	x += 12
	col = x // SPRITE_HEIGHT
	row = (y-0) // SPRITE_WIDTH

	return (col,row)

def get_tile_type(ram,dx,dy,marioPos):
	x = marioPos[POS_X] + dx
	y = marioPos[POS_Y] + dy + SPRITE_HEIGHT

	# Tile locations have two pages. Determine which page we are in
	page = (x // RESOLUTION_X) % 2
	# Figure out where in the page we are
	sub_page_x = (x % RESOLUTION_X) // SPRITE_WIDTH
	sub_page_y = (y - SPRITE_HEIGHT * 2) // SPRITE_HEIGHT  # The PPU is not part of the world, coins, etc (status bar at top)
	if sub_page_y not in range(13):# or sub_page_x not in range(16):
	    return TILE_EMPTY

	addr = 0x500 + page*208 + sub_page_y*16 + sub_page_x
	return ram[addr]

def get_tile_position(x,y):
	return (x//16,y//16-2)

def get_tile(x,y,ram):
	page = (x // RESOLUTION_X) % 2
	sub_x = (x % RESOLUTION_X) // SPRITE_WIDTH
	sub_y = (y - 32) // SPRITE_HEIGHT

	if sub_y not in range(13):
		return TILE_EMPTY

	addr = 0x500 + page*208 + sub_y*SPRITE_HEIGHT + sub_x
	#if ram[addr]: return TILE_FAKE
	if ram[addr] == TILE_EMPTY or ram[addr] == TILE_COIN:
		return TILE_EMPTY
	else:
		return TILE_FAKE

	#return ram[addr]

def get_tiles(ram):
	tiles = {}

	row = 0
	col = 0

	marioPos_level = get_mario_pos_level(ram)
	marioPos_screen = get_mario_pos_screen(ram)

	xStart = marioPos_level[POS_X]-marioPos_screen[POS_X]

	enemies = get_enemy_tile_pos(ram)
	yStart = 0
	marioX,marioY = marioPos_level[POS_X],marioPos_level[POS_Y]
	marioY += SPRITE_HEIGHT

	#MarioX must be within the screen offset
	marioX = ram[PLAYER_POS_X_SCREEN_OFFSET]

	for yPos in range(0,RESOLUTION_Y,SPRITE_HEIGHT):
		for xPos in range(xStart,xStart+RESOLUTION_X,SPRITE_WIDTH):
			pos = (row,col)

			tile = get_tile(xPos,yPos,ram)
			x, y = xPos, yPos
			page = (x // RESOLUTION_X) % 2
			sub_x = (x % RESOLUTION_X) // SPRITE_WIDTH
			sub_y = (y - SPRITE_HEIGHT) // SPRITE_HEIGHT                
			addr = 0x500 + page*208 + sub_y*SPRITE_HEIGHT + sub_x

			#There are no tiles here, since the status bar is there
			if row < 2:
				tiles[pos] = TILE_EMPTY
			else:
				tiles[pos] = tile
				#tiles[pos] = tile

				for enemy in enemies:
					enemyX = enemy[POS_X]
					enemyY = enemy[POS_Y]+SPRITE_HEIGHT//2
					# If the location of the enemy is within 8 tiles of the current tile, put it there
					if abs(xPos-enemyX) <= SPRITE_WIDTH//2 and abs(yPos-enemyY) <= SPRITE_HEIGHT//2:
						#tiles[pos] = enemy[2] # Index 2 is their ID
						tiles[pos] = -1
						#tiles[pos] = 2

			col += 1

		col = 0
		row += 1

	#pos = get_mario_row_col(ram)
	#tiles[pos] = DYNAMIC_MARIO
	#tiles[pos] = -1

	return tiles

def print_tiles(ram):

	tiles = get_tiles(ram)

	for y in range(RESOLUTION_Y//SPRITE_HEIGHT):
		for x in range(RESOLUTION_X//SPRITE_WIDTH-1):
			print(tiles[(x,y)],end=" ")
		print("")

def print_tiles_in_front(ram):

	tiles = get_tiles(ram)
	marioPos = get_mario_row_col(ram)

	for x in range(marioPos[POS_X],marioPos[POS_X]+7):
		for y in range(4,RESOLUTION_Y//SPRITE_HEIGHT-1):
			if x > 15: #For some reason sometimes mario is a bit too far forward
				print("test")
				print(tiles[(y,15)],end=" ")
			else:
				print(tiles[(y,x)],end=" ")
		print("")
	print("")

	for y in range(4, RESOLUTION_Y//SPRITE_HEIGHT-1):
		if marioPos[POS_Y] == y:
			print("1",end=" ")
		else:
			print("0",end=" ")
	print("\n")

FRAMES_POW = 1.5
DIST_POW = 1.8
DIST_TO_END = 3000 #IDK if this is the right distance, I took this from someone else
DIST_EXPLORE = 100

def calc_fitness(frames,dist):
	return max(
		dist ** DIST_POW -
		frames ** FRAMES_POW + 
		(100000 if dist >= DIST_TO_END else 0)+ #Have you reached the "end" yet?
		(2000 if dist >= DIST_EXPLORE else 0), #This should encourage early moving right
		#min(max(dist-50,0),1) * 2500, # Reward early exploration
		0
		)

def calc_fitness_2(dist,time):
	if dist > DIST_TO_END:
		return 100000
	elif dist > DIST_EXPLORE:
		return dist + 0.1*(400-time)
	else:
		return dist
