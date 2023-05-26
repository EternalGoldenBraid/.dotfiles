import pygame
import os

# Initialize Pygame
pygame.init()

# Define screen dimensions and create the screen
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Set the title of the window
pygame.display.set_caption("Flying Armbar from a Handstand")

# Define colors
background_color = (255, 255, 255)
character_color = (0, 0, 255)
opponent_color = (255, 0, 0)

# Load character images
base_path = 'path/to/images/'
character_img = pygame.image.load(os.path.join(base_path, 'character.png'))
opponent_img = pygame.image.load(os.path.join(base_path, 'opponent.png'))

# Scale images to an appropriate size
character_img = pygame.transform.scale(character_img, (100, 100))
opponent_img = pygame.transform.scale(opponent_img, (100, 100))

def draw_characters():
    screen.blit(character_img, (350, 400))
    screen.blit(opponent_img, (450, 450))

def main():
    running = True

    while running:
        # Fill the screen with the background color
        screen.fill(background_color)

        # Draw the characters
        draw_characters()

        # Update the screen
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Quit Pygame
    pygame.quit()

if __name__ == '__main__':
    main()
