def nearest_square(y):
        """ 
        Takes an integer argument limit, and returns the largest square number 
        that is less than limit. A square number is the product of an 
        integer multiplied by itself, for example 36 is a square number because 
        it equals 6*6.
        """
        x = 0  
        while( x ** 2 <= y ): x += 1
        if x >= 1: return (x - 1) ** 2
        else: return 0

"""
Another solution
def nearest_square(limit):
        answer = 0
            while (answer+1)**2 < limit:
                    answer += 1
                        return answer**2
"""

"""
Build a set of all of the square numbers greater than 0 and less than 2,000.
"""
squares = set()
i = 1
while i < 2000:
    squares.add(nearest_square(i))
    i = i + 1

print(squares)

