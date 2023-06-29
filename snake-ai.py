import pygame
import random
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam




class SnakeAI :

    def __init__(self) :

        pass


    def reset(self, n, show = False) :

        self.n = n

        if show :

            pygame.init()
        
            self.zoom = 10
            self.width  = self.n * self.zoom
            self.height = self.n * self.zoom

            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake')

            self.clock = pygame.time.Clock()
            self.fps = 20

        self.n = n

        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.snake = [(self.n // 2, self.n // 2), (self.n // 2 - 1, self.n // 2)]

        self.food = self.getFood()

        self.score = 0

        self.color = (255, 255, 0)
        

        stateHead, stateBody, stateFood = np.zeros((self.n, self.n)), np.zeros((self.n, self.n)), np.zeros((self.n, self.n))
        stateHead[self.snake[0]] = 1
        stateFood[self.food] = 1
        self.state = np.append(stateBody.flatten(), stateHead.flatten())
        self.state = np.append(self.state, stateFood.flatten())
        self.state = np.append(self.state, 1*(self.direction == np.array(['UP', 'DOWN', 'LEFT', 'RIGHT'])))

        self.gameOver = False


    def getFood(self) :
        
        xFood = random.randrange(0, self.n)
        yFood = random.randrange(0, self.n)

        return (int(xFood), int(yFood))

    def drawSnake(self) :

        for part in self.snake:
            pygame.draw.rect(self.display, self.color, (self.zoom * part[0], self.zoom * part[1], 10, 10))

    def drawFood(self) :

        pygame.draw.rect(self.display, (252, 57, 3), (self.zoom * self.food[0], self.zoom * self.food[1], 10, 10))

    def showScore(self):
        font = pygame.font.Font(None, 24)
        score_text = font.render('Score: ' + str(self.score), True, (255, 255, 255))
        self.display.blit(score_text, (10, self.height - 10))

    def chooseAction(self, model = None, epsilon = 1.1) :

        # move encoder :  [STRAIGHT; LEFT TURN; RIGHT TURN]

        if np.random.rand() < epsilon :
            chosen = np.random.randint(3)    

        else :
            Qvalues = model.predict(self.state[np.newaxis], verbose = 0)
            chosen = np.argmax(Qvalues)

        self.moveName = ['STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT'][chosen]
        move = np.zeros(3)
        move[chosen] = 1

        return move


    def performAction(self) :

        reward = 0

        x, y = self.snake[0]
        xf, yf = self.food[0], self.food[1]
        
        if (self.direction == 'UP' and self.moveName == 'STRAIGHT') or (self.direction == 'RIGHT' and self.moveName == 'TURN_LEFT') or (self.direction == 'LEFT' and self.moveName == 'TURN_RIGHT'):
            y -= 1
            self.direction = 'UP'
        elif (self.direction == 'DOWN' and self.moveName == 'STRAIGHT') or (self.direction == 'RIGHT' and self.moveName == 'TURN_RIGHT') or (self.direction == 'LEFT' and self.moveName == 'TURN_LEFT'):
            y += 1
            self.direction = 'DOWN'
        elif (self.direction == 'LEFT' and self.moveName == 'STRAIGHT') or (self.direction == 'UP' and self.moveName == 'TURN_LEFT') or (self.direction == 'DOWN' and self.moveName == 'TURN_RIGHT'):
            x -= 1
            self.direction = 'LEFT'
        elif (self.direction == 'RIGHT' and self.moveName == 'STRAIGHT') or  (self.direction == 'UP' and self.moveName == 'TURN_RIGHT') or (self.direction == 'DOWN' and self.moveName == 'TURN_LEFT'):
            x += 1
            self.direction = 'RIGHT'

        new_head = (x, y)
        self.snake.insert(0, new_head)
        if new_head == self.food :
            self.food = self.getFood()
            self.score += 1
            reward = 100
        else :
            self.snake.pop()

        # Distance-based reward
        d = np.sqrt((x - xf) ** 2 + (y - yf) ** 2)
        reward += np.exp(-5*d / self.n)
        self.color = (255, 255, (d / (np.sqrt(2)*self.n)) ** 2 * 255)

        if x < 0 or x >= self.n or y < 0 or y >= self.n :
            self.gameOver = True
            reward = -100

        if new_head in self.snake[1:] :
            self.gameOver = True
            reward = -100

        return reward

    def updateState(self) :

        if not self.gameOver :
            stateHead, stateBody, stateFood = np.zeros((self.n, self.n)), np.zeros((self.n, self.n)), np.zeros((self.n, self.n))
            stateHead[self.snake[0]] = 1
            stateFood[self.food] = 1
            for part in self.snake[1:] :
                stateBody[part] = 1
            self.state = np.append(stateBody.flatten(), stateHead.flatten())
            self.state = np.append(self.state, stateFood.flatten())
            self.state = np.append(self.state, 1*(self.direction == np.array(['UP', 'DOWN', 'LEFT', 'RIGHT'])))

        else :
            pass

    def play(self):

        while not self.gameOver :

            self.display.fill((0, 0, 0))
            self.updateState()
            self.drawSnake()
            self.drawFood()
            self.chooseAction()
            print(self.direction, self.move)
            self.performAction()
            self.showScore()

            pygame.display.update()
            self.clock.tick(self.fps)

        pygame.quit()

class DQN :

    def __init__(self, n) :

        self.n = n

        # Parameters
        self.epsilon = 1.0
        self.decay = 0.99
        self.min_epsilon = 0.01
        self.gamma = 0.99
        self.learningRate = 0.001
        self.batchSize = 32
        self.episodes = 200

    def buildModel(self) :

        # Define the neural network model
        self.model = Sequential()
        self.model.add(Dense(256, input_shape = (3*(self.n ** 2) + 4,), activation = 'relu'))
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dense(64, activation = 'relu'))
        self.model.add(Dense(3, activation = 'softmax'))

        # Compile the model
        self.model.compile(optimizer = Adam(learning_rate = self.learningRate), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    def train(self, memory) :

        for state, move, reward, next_state, done in memory :

            target = reward

            if not done :

                nextQvalues = self.model.predict(next_state.reshape(1,-1), verbose = 0)
                target += self.gamma * np.max(nextQvalues)

            Qvalues = self.model.predict(state.reshape(1,-1), verbose = 0)
            Qvalues[0][np.argmax(move)] = target
            self.model.fit(state.reshape(1,-1), Qvalues.reshape(1,-1), verbose = 0)

    def Qlearning(self, show = False) :

        SNAKE = SnakeAI()


        for episode in range(self.episodes) :
            
          #  SNAKE.gaming()
            SNAKE.reset(self.n, show = show)
            memory = []
            while not SNAKE.gameOver :

                state  = SNAKE.state
                if show :
                    SNAKE.display.fill((0, 0, 0))
                    SNAKE.drawFood()
                    SNAKE.drawSnake()
                    SNAKE.showScore()
                move   = SNAKE.chooseAction(self.model, self.epsilon)
                reward = SNAKE.performAction()
                SNAKE.updateState()
                next_state = SNAKE.state
                done = SNAKE.gameOver
                if show :
                    pygame.display.update()
                    SNAKE.clock.tick(SNAKE.fps)

                memory.append([state, move, reward, next_state, done])


            print('\n')
            totalReward = np.sum([ep[2] for ep in memory])
            print(f'Episode {episode} :  score {SNAKE.score}, no. moves {len(memory)}, rewards {round(totalReward)}, epsilon {round(self.epsilon, 2)}')


            self.train(memory)

            if self.epsilon > self.min_epsilon :
                self.epsilon = self.epsilon * self.decay


snakey = DQN(30)
snakey.buildModel()
snakey.Qlearning(show = True)