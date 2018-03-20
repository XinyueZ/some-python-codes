def most_prolific(dict):
    """ 
    Takes a dict formatted like "book->published year"
    
    {"Please Please Me": 1963, "With the Beatles": 1963, 
        "A Hard Day's Night": 1964, "Beatles for Sale": 1964, "Twist and Shout": 1964,
            "Help": 1965, "Rubber Soul": 1965, "Revolver": 1966,
                "Sgt. Pepper's Lonely Hearts Club Band": 1967,
                    "Magical Mystery Tour": 1967, "The Beatles": 1968,
                        "Yellow Submarine": 1969 ,'Abbey Road': 1969,
                            "Let It Be": 1970}

    and returns the year in which the most albums were released. 
    If you call the function on the Beatles_Discography it should return 1964, 
    which saw more releases than any other year in the discography.
    If there are multiple years with the same maximum number of releases, 
    the function should return a list of years.
    """
    
    # Make map: value -> count
    # For example: 1963 -> 3, 1963 -> 2 .
    value_counts = {}
    for key in dict:
        value = dict[key]
        current_count = value_counts.get(value, 0)
        current_count += 1
        value_counts[value] = current_count

    # Make map: count -> list of key
    # For example: 3 -> {1964}, 2 -> {1963, 1969, 1965, 1967}
    count_rankings = {}
    for key in value_counts:
        count = value_counts[key]  
        ranking_bucket = count_rankings.get(count, [])
        ranking_bucket.append(key)
        count_rankings[count] = ranking_bucket

    max_count = sorted(count_rankings).pop()
    result_list = count_rankings[max_count]
    if len(result_list) > 1: return result_list
    else: return result_list[0]

Beatles_Discography = {"Please Please Me": 1963, "With the Beatles": 1963, 
            "A Hard Day's Night": 1964, "Beatles for Sale": 1964, "Twist and Shout": 1964,
                "Help": 1965, "Rubber Soul": 1965, "Revolver": 1966,
                    "Sgt. Pepper's Lonely Hearts Club Band": 1967,
                        "Magical Mystery Tour": 1967, "The Beatles": 1968,
                            "Yellow Submarine": 1969 ,'Abbey Road': 1969,
                                "Let It Be": 1970}
print(most_prolific(Beatles_Discography))


Beatles_Discography = {'The Game': 1980, 'A Night at the Opera': 1975, 'Jazz': 1978, 'Queen II': 1974, 'A Day at the Races': 1976, 'News of the World': 1977, 'Queen': 1973, 'Sheer Heart Attack': 1974}
print(most_prolific(Beatles_Discography))

