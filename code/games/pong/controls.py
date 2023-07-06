#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2022
Creation: 06.07.23
Description: Control objects of the screen
"""

import pygame
class Paddle:
    def __init__(self, paddle_w, paddle_h, screen, speed=10):
        """
        Paddle class
        Args:
            paddle_w: paddle width, in pixels
            paddle_h: paddle height, in pixels
            screen_w: screen width, in pixels
            screen_h: screen height, in pixels
            speed: paddle speed, in pixels
        """
        self.x = 0
        self.y = screen.get_height() // 2
        self.w = paddle_w
        self.h = paddle_h
        self.screen = screen
        self.screen_h = screen.get_height()
        self.screen_w = screen.get_width()
        self.speed = speed


    def move(self, direction):
        if direction == 1 and self.y > 0:
            self.y -= self.speed
        elif direction == -1 and self.y < self.screen_h - self.h:
            self.y += self.speed

    def draw(self, color=(255, 255, 255)):
        pygame.draw.rect(self.screen, color, pygame.Rect(self.x, self.y, self.w, self.h))



class Ball:
    def __init__(self, screen, dx=1, dy=1, ball_size=40):
        """
        Ball class
        Args:
            dx: velocity in x direction
            dy: velocity in y direction
            screen_w: screen width, in pixels
            screen_h: screen height, in pixels
            ball_size: ball size, in pixels
        """
        self.x = screen.get_width() // 2
        self.y = screen.get_height() // 2
        self.speed = 3
        self.dx = dx
        self.dy = dy
        self.screen = screen
        self.screen_h = screen.get_height()
        self.screen_w = screen.get_width()
        self.ball_size = ball_size

    def move(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed
        if self.y < 0 or self.y > self.screen_h - self.ball_size:
            self.dy *= -1
        if self.x > self.screen_h - self.ball_size:  # Adding bouncing for the third border
            self.dx *= -1

    def draw(self, color=(255, 255, 255)):
        pygame.draw.circle(self.screen, color, (self.x, self.y), self.ball_size // 2)

    def check_collision(self, paddle):
        if self.dx < 0 and self.x < paddle.x + paddle.w and paddle.y < self.y < paddle.y + paddle.h:
            self.dx *= -1
            return True
        return False