"""
Use python turtle draw api to draw some shapes.
"""
import turtle


def draw_square(paint):
    for _ in range(4):
        paint.forward(100)
        paint.right(90)
    return paint

def draw_shapes():
    window = turtle.Screen()
    window.bgcolor("red")

    paint = turtle.Turtle()
    paint.color("yellow")
    for _ in range(360):
        draw_square(paint).right(1)

    window.exitonclick()

draw_shapes()
