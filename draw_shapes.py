import turtle

def draw_shapes():
    window = turtle.Screen()
    window.bgcolor("red")
    paint = turtle.Turtle()
    paint.forward(100)
    paint.right(90)
    
    paint.forward(100)
    paint.right(90)

    paint.forward(100)
    paint.right(90)

    paint.forward(100)
    paint.right(90)

    drawer = turtle.Turtle()
    drawer.shape("arrow")
    drawer.color("yellow")
    drawer.circle(40)

    window.exitonclick()

draw_shapes()
