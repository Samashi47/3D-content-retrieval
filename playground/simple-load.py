from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from objloader import OBJ
import sys

rotate_x, rotate_y = 0, 0
translate_x, translate_y = 0, 0
zoom = 5
last_x, last_y = 0, 0
left_button = False
right_button = False

def init():
    glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    
    global obj
    obj = OBJ(sys.argv[1], swapyz=True)
    print(f"Loaded '{sys.argv[1]}' with {len(obj.vertices)} vertices and {len(obj.faces)} faces")

def resize(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(90.0, width/float(height), 1, 1000.0)
    glMatrixMode(GL_MODELVIEW)

def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glTranslatef(translate_x/20., translate_y/20., -zoom)
    glRotatef(rotate_y, 1, 0, 0)
    glRotatef(rotate_x, 0, 1, 0)
    glCallList(obj.gl_list)

    glutSwapBuffers()

def mouse(button, state, x, y):
    global left_button, right_button, last_x, last_y, zoom
    
    last_x, last_y = x, y
    
    if button == GLUT_LEFT_BUTTON:
        left_button = (state == GLUT_DOWN)
    elif button == GLUT_RIGHT_BUTTON:
        right_button = (state == GLUT_DOWN)
    elif button == 3:  # Mouse wheel up
        zoom = max(1, zoom-0.2)
        glutPostRedisplay()
    elif button == 4:  # Mouse wheel down
        zoom += 0.2
        glutPostRedisplay()

def motion(x, y):
    global rotate_x, rotate_y, translate_x, translate_y, last_x, last_y
    
    dx = x - last_x
    dy = y - last_y
    
    if left_button:
        rotate_x += dx
        rotate_y += dy
    elif right_button:
        translate_x += dx
        translate_y -= dy
        
    last_x, last_y = x, y
    glutPostRedisplay()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"OpenGL Model Viewer")

    init()
    
    glutDisplayFunc(draw)
    glutReshapeFunc(resize)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    
    glutMainLoop()

if __name__ == "__main__":
    main()