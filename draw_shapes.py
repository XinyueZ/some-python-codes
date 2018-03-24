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



def draw_shapes():
    window = turtle.Screen()
    window.bgcolor("red")

    draw_square()
    draw_circle()

    window.exitonclick()

draw_shapes()
