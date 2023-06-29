import pygame
import random
import numpy as np
from keras import Sequential
from keras.layers import Dense




class Snake :

    def __init__(self) :

        pygame.init()
        
        self.width  = 500
        self.height = 500
        self.zoom = 10

        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')

        self.clock = pygame.time.Clock()
        self.fps = 10

        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.snake = [(self.width // self.zoom // 2, self.height // self.zoom // 2)]

        self.food = self.getFood()

        self.score = 0
        print(self.snake[0])

        self.state = np.zeros((self.width // self.zoom, self.height // self.zoom))
        self.state[self.snake[0]] = 2
        self.state[self.food] = 3
        self.state = np.append(self.state.flatten(), 1*(self.direction == np.array(['UP', 'DOWN', 'LEFT', 'RIGHT'])))

        self.gameOver = False


    def getFood(self) :
        
        xFood = random.randrange(0,  self.width / self.zoom)
        yFood = random.randrange(0, self.height / self.zoom)

        return (int(xFood), int(yFood))

    def drawSnake(self) :

        for part in self.snake:
            pygame.draw.rect(self.display, (186, 252, 3), (self.zoom * part[0], self.zoom * part[1], 10, 10))

    def drawFood(self) :

        pygame.draw.rect(self.display, (252, 57, 3), (self.zoom * self.food[0], self.zoom * self.food[1], 10, 10))

    def showScore(self):
        font = pygame.font.Font(None, 24)
        score_text = font.render('Score: ' + str(self.score), True, (255, 0, 0))
        self.display.blit(score_text, (self.width / 2, self.height / 2))


    def moveSnake(self) :

        x, y = self.snake[0]
        
        if self.direction == 'UP' :
            y -= 1
        elif self.direction == 'DOWN' :
            y += 1
        elif self.direction == 'LEFT' :
            x -= 1
        elif self.direction == 'RIGHT' :
            x += 1

        new_head = (x, y)
        self.snake.insert(0, new_head)
        if new_head == self.food :
            self.food = self.getFood()
            self.score += 1
        else :
            self.snake.pop()

        if x < 0 or x >= self.width / self.zoom or y < 0 or y >= self.height / self.zoom :
            self.gameOver = True

        if new_head in self.snake[1:] :
            self.gameOver = True

      #  self.action = 1*(self.direction == np.array(['UP', 'DOWN', 'LEFT', 'RIGHT']))

    def userMove(self, move) :

        if (
            (move == 'UP' and self.direction != 'DOWN')
            or (move == 'DOWN' and self.direction != 'UP')
            or (move == "LEFT" and self.direction != 'RIGHT')
            or (move == 'RIGHT' and self.direction != "LEFT")
        ):
            self.direction = move

    def updateState(self) :

        self.state = np.zeros((self.width // self.zoom, self.height // self.zoom))
        self.state[self.snake[0]] = 2
        self.state[self.food] = 3
        for part in self.snake[1:] :
            self.state[part] = 1
        self.state = np.append(self.state.flatten(), 1*(self.direction == np.array(['UP', 'DOWN', 'LEFT', 'RIGHT'])))


    def play(self):

        while not self.gameOver :

            for event in pygame.event.get() :

                if event.type == pygame.QUIT :
                    self.game_over = True

                elif event.type == pygame.KEYDOWN :

                    if event.key == pygame.K_UP :
                        self.userMove('UP')

                    elif event.key == pygame.K_DOWN :
                        self.userMove('DOWN')

                    elif event.key == pygame.K_LEFT :
                        self.userMove('LEFT')

                    elif event.key == pygame.K_RIGHT :
                        self.userMove('RIGHT')

            self.display.fill((0, 0, 0))
            self.drawSnake()
            self.drawFood()
            self.moveSnake()
            self.showScore()
            self.updateState()

            pygame.display.update()
            self.clock.tick(self.fps)

        pygame.quit()

class DQN :

    def __init__(self) :

        height, width = 500, 500
        zoom = 10
        n_state = (height / zoom) * (width / zoom)
        
        # Define the neural network model
        self.model = Sequential()
        self.model.add(Dense(256, input_shape = (n_state,) , activation = 'relu'))
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dense(4, activation = 'softmax'))  # 4 possible actions: up, down, left, right

    def chooseAction(self) :

        pass





game = Snake()
game.play()
