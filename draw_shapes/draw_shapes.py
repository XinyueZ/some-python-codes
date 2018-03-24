"""
Use python turtle draw api to draw some shapes.
"""
import turtle

def draw_square():
    paint = turtle.Turtle()
    for _ in range(4):
        paint.forward(100)
        paint.right(90)

def draw_circle():
    drawer = turtle.Turtle()
    drawer.shape("arrow")
    drawer.color("yellow")
    drawer.circle(40)

def draw_triangle():
    paint = turtle.Turtle()
    paint.color("blue")
    angle = 120
    for _ in range(3):
        paint.right(angle)
        paint.forward(100)



def draw_shapes():
    window = turtle.Screen()
    window.bgcolor("red")

    draw_square()
    draw_circle()
    draw_triangle()

    window.exitonclick()

draw_shapes()
