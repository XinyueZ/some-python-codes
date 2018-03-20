def hours2days(period):
    """
    uses a tuple to return multiple values. Write an hours2days function that takes one argument, an integer, that is a time period in hours. The function should return a tuple of how long that period is in days and hours, with hours being the remainder that can't be expressed in days. For example, 39 hours is 1 day and 15 hours, so the function should return (1,15).

    These examples demonstrate how the function can be used:
    
    hours2days(24) # 24 hours is one day and zero hours
    (1, 0)
    hours2days(25) # 25 hours is one day and one hour
    (1, 1)
    hours2days(10000)
    (416, 16) 
    """
    hours_of_day = 24
    day = period // hours_of_day
    hours = period % hours_of_day
    return day, hours

print(hours2days(24) == (1, 0))
print(hours2days(25) == (1, 1))
print(hours2days(10000) == (416, 16))
