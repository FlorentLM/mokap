import sys

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtGui import QSurfaceFormat, QOpenGLContext

import OpenGL.GL as gl
import OpenGL.GLU as glu


class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._angle_x = 0
        self._angle_y = 0
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._zoom = -5.0

        self._mouse_last_pos = QPoint()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_view)
        self.timer.start(16)

        self.elements = []

        cube = {
            'mode': 'lines',
            'vertices': [
                # Top
                [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],    # RB, LB
                [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0],      # LF, RF

                # Bottom
                [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0],    # RF, LF
                [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0]   # LB, RB
            ],
            'edges': [
                [0, 1], [1, 2], [2, 3], [3, 0],         # Top
                [4, 5], [5, 6], [6, 7], [7, 4],         # Bottom
                [0, 7], [1, 6], [2, 5], [3, 4],         # Side connections
            ]
        }

        points = {
            'mode': 'points',
            'vertices': cube['vertices'],
            'size': 10.0,
            'colour': (0.3, 0.8, 0.8),
        }

        sphere = {
            'mode': 'sphere',
            'centre': [0.0, 0.0, 0.0],
            'radius': 0.1,
            'colour': (0.0, 1.0, 0.0)
        }

        self.elements.append(cube)
        self.elements.append(points)
        # self.elements.append(sphere)

    def initializeGL(self):
        self.context = QOpenGLContext(self)

        # Create an OpenGL 2.1 context - becaue we don't need any shaders so fixed pipeline is best
        format = QSurfaceFormat()
        format.setVersion(2, 1)
        format.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
        self.context.setFormat(format)
        if not self.context.create():
            raise Exception("Unable to create GL context")

        gl.glClearColor(0.5, 0.8, 0.7, 1.0)        # Background colour
        gl.glEnable(gl.GL_DEPTH_TEST)                                     # Depth test is needed
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)        # Enable transparency
        gl.glEnable(gl.GL_BLEND)                                          # Blend transparency
        gl.glEnable(gl.GL_POINT_SMOOTH)                                   # Round points

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45.0, w / h, 1.0, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # move and rotate the GL camera according to user input
        gl.glTranslatef(self._pos_x, self._pos_y, self._zoom)
        gl.glRotatef(self._angle_x, 1.0, 0.0, 0.0)
        gl.glRotatef(self._angle_y, 0.0, 1.0, 0.0)

        # and finally draw all the stuff
        self.draw_elements()

    def draw_elements(self):

        for obj in self.elements:

            match obj['mode']:
                case 'lines':
                    gl.glBegin(gl.GL_LINES)
                    gl.glColor3f(*obj.get('colour', (1.0, 1.0, 1.0)))
                    for e in obj['edges']:
                        for v in e:
                            gl.glVertex3fv(obj['vertices'][v])
                    gl.glEnd()

                case 'points':
                    gl.glPointSize(obj.get('size', 1.0))
                    gl.glColor3f(*obj.get('colour', (1.0, 1.0, 1.0)))
                    gl.glBegin(gl.GL_POINTS)
                    for v in obj['vertices']:
                        gl.glVertex3f(*v)
                    gl.glEnd()

                case 'sphere':
                    gl.glColor3f(*obj.get('colour', (1.0, 1.0, 1.0)))
                    gl.glPushMatrix()
                    gl.glTranslatef(*obj['centre'])
                    gluQuadric = glu.gluNewQuadric()
                    glu.gluSphere(gluQuadric, obj.get('radius', 1.0), 64, 64)
                    glu.gluDeleteQuadric(gluQuadric)
                    gl.glPopMatrix()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_last_pos = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._mouse_last_pos
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._pos_x += delta.x() * 0.01
                self._pos_y -= delta.y() * 0.01
            else:
                self._angle_x += delta.y() * 0.5
                self._angle_y += delta.x() * 0.5
            self._mouse_last_pos = event.position()
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self._zoom += delta * 0.01
        self.update()

    def _update_view(self):
        self.update()


class MinimalWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Minimal OpenGL window")
        self.setGeometry(100, 100, 800, 600)

        self.opengl_widget = OpenGLWidget(self)
        self.setCentralWidget(self.opengl_widget)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MinimalWindow()
    window.show()

    sys.exit(app.exec())
