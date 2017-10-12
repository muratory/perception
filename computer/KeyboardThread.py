import threading
import Queue
from commonDeepDriveDefine import *
from commonDeepDriveTools import *
import pygame
from pygame.locals import *

WINDOW_SIZE=(200,200)

class keyboardThread(commonThread):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(keyboardThread, self).__init__()


    def run(self):
        # use the win
        pygame.init()
        pygame.display.set_caption(self.name)
        pygame.font.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.font =pygame.font.SysFont(None, 20)
        super(keyboardThread, self).run()

    def displayText(self,text=None):
        #display and return key
        if text == None:
            pygameText=self.font.render('No handle for key '+str(self.strKey), 1,(255,0,0))

        else:
            pygameText=self.font.render(text, 1,(255,255,255))
        
        self.screen.blit(pygameText, (5, self.textYpos))
        self.textYpos+=10
        if self.textYpos > WINDOW_SIZE[1] :
            self.screen.fill((0, 0, 0))
            self.textYpos=0
        
        pygame.display.flip()
    
    def _handle_RECEIVE(self, cmd):
        self.textYpos = 0
        while True:
            #check first if new command to stop comes in
            try:
                newCmd = self.cmd_q.get(False)
                if newCmd.type == ClientCommand.STOP:
                    self._handle_STOP(cmd)
                    return
            except Queue.Empty as message:
                #we should always be there
                pass
            
            try:
                # receive new input from human driver
                for event in pygame.event.get():
                    self.strKey = ''
                    if event.type == KEYDOWN:
                        if event.key == K_RIGHT:
                            self.strKey = 'right'

                        elif event.key == K_LEFT:
                            self.strKey = 'left'

                        elif event.key == K_SPACE:
                            self.strKey = 'space'

                        elif event.key == K_DOWN:
                            self.strKey = 'down'

                        elif event.key == K_UP:
                            self.strKey = 'up'
                            
                        elif event.key == K_r:
                            self.strKey = 'RIGHT_TURN'
                            
                        elif event.key == K_l:
                            self.strKey = 'LEFT_TURN'   
                            
                        elif event.key == K_i:
                            self.strKey = 'IDLE'
                            
                        elif event.key == K_s:
                            self.strKey = 'STRAIGHT'
                            
                        elif event.key == K_n:
                            self.strKey = 'NN_CONTROL'
                            
                        elif event.key == K_g:
                            self.strKey = 'GPS'                    

                        elif event.key == K_KP_PLUS or event.key == K_PLUS or event.key == K_EQUALS :
                            self.strKey = 'plus'
                            
                        elif event.key == K_KP_MINUS or event.key == K_MINUS or event.key == 54:
                            self.strKey = 'minus'
                            
                        elif event.key == K_KP1:
                            self.strKey = 'ONE'
                            
                        elif event.key == K_KP2:
                            self.strKey = 'TWO'
                            
                        elif event.key == K_ESCAPE or event.key == K_q or event.key == K_a :
                            self.strKey = 'exit'
                               
                        elif event.key == K_p:
                            self.strKey = 'PATH_CONTROL'
                            
                        elif event.key == K_h:
                            self.strKey = 'help'

                        elif event.key == K_m:
                            self.strKey = 'MAP'
                            
                        else :
                            #this key is probably unknown but return it and let main decide
                            self.strKey = str(event.key)

                        #display and return key
                        self.reply_q.put(self._success_reply(self.strKey))

                            
                    elif event.type == KEYUP:
                        key_input = pygame.key.get_pressed()
                        #when keyup, only test if all keys are UP
                        if ((key_input[pygame.K_RIGHT] == 0) and
                        (key_input[pygame.K_LEFT] == 0) and
                        (key_input[pygame.K_UP] == 0) and
                        (key_input[pygame.K_DOWN] == 0) and
                        (key_input[pygame.K_SPACE] == 0)):
                            self.reply_q.put(self._success_reply('none'))
                            
            except IOError as e:
                pass
                #print 'Steer Connect Error : ' + str(e)
                #retry again

